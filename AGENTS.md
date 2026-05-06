# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

Research code accompanying the paper ["Variational Data Assimilation with a Learned Inverse Observation Operator"](https://arxiv.org/abs/2102.11192) (Frerix et al., 2021). Two dynamical systems are implemented: **Lorenz96** (1D) and **KolmogorovFlow** (2D incompressible Navier–Stokes with Kolmogorov forcing, via JAX-CFD).

## Environment

Pinned legacy stack — do not attempt to upgrade casually:
- `jax==0.2.6`, `jaxlib==0.1.57` (CUDA-specific wheel, e.g. `jaxlib==0.1.57+cuda110`)
- `flax==0.3.0` (uses the deprecated `flax.nn.Module` / `init_by_shape` / `flax.optim` API — this is NOT the modern Linen API)
- `jax_cfd==0.1.0` (old API: `cfd.grids.AlignedArray`, `cfd.grids.Grid(..., domain=...)`, `cfd.equations.stable_time_step`, `cfd.forcings.simple_turbulence_forcing`)
- Other deps in `requirements.txt`: numpy, scipy, xarray, seaborn

Before modifying ML / dynamics code, assume the imports are correct and match these pinned versions — don't "fix" them to current Flax/JAX-CFD conventions.

## Common commands

All pipeline scripts take a single `--config` argument pointing to a JSON file under `config_files/`.

```bash
# Data assimilation (main experiment)
python run_data_assimilation.py --config config_files/data_assimilation/lorenz96_baselineinit_obsopt.config

# Generate trajectories for training the inverse observation operator
python run_generate_training_data.py --config config_files/data_generation/lorenz96_generate_data.config

# Train the inverse observation model
python run_train_inverse_observations.py --config config_files/invobs_training/lorenz96_train_invobs.config

# Precompute spatial correlation (C^{1/2}, C^{-1/2}) used to precondition DA
python run_compute_correlation.py --config config_files/correlation/lorenz96_correlation.config

# Tests (pytest)
pytest tests/
pytest tests/test_data_assimilation.py::test_correlation_transform
```

Config files hard-code paths under `/data/invobs-da-data/` (pretrained models + correlation files downloaded via `gsutil cp -r gs://gresearch/jax-cfd/projects/invobs-data-assimilation/invobs-da-data /data`). `config_files/test/` contains smaller configs used by the pytest suite.

Note: `tests/test_data_assimilation.py` imports `from data_assimilation import ...` but the module file is `run_data_assimilation.py` — the tests as written will not import unless run from an environment where that name is aliased. Flag this rather than silently renaming.

## Architecture

The codebase is organized around **four pipeline stages** that share a common `DynamicalSystem` abstraction:

1. **Data generation** (`run_generate_training_data.py`) — integrates a dynamical system forward to produce `(state, observation)` trajectory pairs.
2. **Inverse observation training** (`run_train_inverse_observations.py`) — trains a Flax CNN (`ObservationInverterLorenz96` / `ObservationInverterKolmogorov`) that maps observation sequences `Y` back to physics space `X`. Output is a pickled `{'model_state': ...}`.
3. **Correlation computation** (`run_compute_correlation.py`) — estimates the spatial covariance `C` of the system's stationary distribution and stores `C^{1/2}` and `C^{-1/2}` as an xarray netCDF. These are used to **precondition** the DA optimization (optimize in decorrelated coordinates `z = C^{-1/2} x`).
4. **Data assimilation** (`run_data_assimilation.py`) — the main experiment. Given observations `Y`, finds `x_0` minimizing a mean-squared trajectory loss via SciPy's L-BFGS-B.

### The DA loss (key insight from the paper)

`da_methods.da_loss_fn` is a single parameterized loss that supports **two spaces**:
- **Observation-space loss**: `|| H(integrate(x_0)) - y ||^2` — the standard 4D-Var objective.
- **Physics-space loss**: `|| integrate(x_0) - H^{-1}_{learned}(y) ||^2` — uses the trained inverse observation model to lift `y` into physics space, avoiding bad local minima when `H` is lossy (sparse observations).
- **Hybrid**: run physics-space L-BFGS first, then observation-space L-BFGS. This is the paper's main contribution.

`run_data_assimilation.py::generate_loss_functions` builds both via `functools.partial` over `da_loss_fn` with different `physics_transform` / `observation_transform` slots. `optimize_da` runs them sequentially; `physics_space_opt_steps` and `obs_space_opt_steps` in the config control which mode is active.

### The DynamicalSystem abstraction (`dynamical_system.py`)

Base class provides `integrate`, `observe`, `batch_integrate`, `batch_warmup`. Concrete systems:
- **`Lorenz96`**: state shape `(grid_size,)`, integrated with `jax.experimental.ode.odeint`. `observe(x) = x[..., ::observe_every]`.
- **`KolmogorovFlow`**: state shape `(grid_size, grid_size, 2)` (velocity components stacked on last axis), integrated with jax-cfd's Van Leer advection + fast-diagonal pressure solve + `jax.checkpoint` for memory. Interop between ndarray and `cfd.grids.AlignedArray` goes through `util.jnp_to_aa_tuple` / `aa_tuple_to_jnp` — the offset list is system state.

`generate_dyn_sys(config)` dispatches on `config['dyn_sys']` (`'lorenz96'` or `'kolmogorov'`). Both DA init strategies (`'baseline'` interpolation/average and `'invobs'` learned) and data generation (`generate_data_lorenz96` / `generate_data_kolmogorov`) also dispatch on this string — adding a new system means touching every dispatch site.

### Correlation preconditioning

`run_data_assimilation.py::generate_correlation_transform` loads the precomputed `cov_sqrt` / `cov_inv_sqrt` and returns a single closure `correlation_transform(x, mode)` where `mode ∈ {'cor', 'dec'}`. Optimization happens in decorrelated `z`-space; `X0_init` is decorrelated before L-BFGS and `Z0_opt` is re-correlated back to physics coordinates at the end.

### File layout (non-obvious pieces)

- `lorenz96_methods.py` / `kolmogorov_methods.py`: system-specific data generation, DA init baselines (averaging / interpolation), spatial helpers.
- `lorenz96_ml.py` / `kolmogorov_ml.py`: Flax modules for the inverse observation networks.
- `ml_methods.py`: shared Flax plumbing — `create_model`, `load_model` (reconstructs the module then deserializes pickled state dict), `create_adam_optimizer`.
- `analysis_util.py`: plotting / loading helpers used exclusively by `Analysis_Lorenz96.ipynb` and `Analysis_KolmogorovFlow.ipynb` (paper figures).

### XLA / determinism

`run_data_assimilation.py` sets `XLA_FLAGS='--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0'` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` at import time. Keep these if modifying the entry point — they're for reproducibility and to avoid OOM on V100.
