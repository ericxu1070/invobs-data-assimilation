# Original JAX pipeline

`paper_scripts/` holds the JAX and Flax code of Frerix et al. (2021). The PyTorch port in the repository root does not import it. This page describes how the scripts fit together and how to run them. The scripts have not been run or tested as part of this repository's capstone work.

The code depends on a pinned legacy stack, so install it in its own environment:

- `jax==0.2.6` and `flax==0.3.0`, both pinned in `requirements.txt`.
- `jaxlib==0.1.57` built for your CUDA version, for example `pip install -U jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html`.
- `jax_cfd==0.1.0` (`pip install -U jax_cfd==0.1.0`), which solves the Kolmogorov flow.
- `numpy`, `scipy`, `xarray`, `seaborn`, also in `requirements.txt`.

`flax==0.3.0` uses the old `flax.nn` API, so do not upgrade it without porting the model code.

## Pipeline stages

Every script takes one `--config` argument that points to a file under `config_files/`. Run the scripts from the repository root.

| Stage | Script | Config folder |
|---|---|---|
| Generate trajectories | `paper_scripts/run_generate_training_data.py` | `config_files/data_generation/` |
| Train the inverse observation model | `paper_scripts/run_train_inverse_observations.py` | `config_files/invobs_training/` |
| Compute spatial correlations | `paper_scripts/run_compute_correlation.py` | `config_files/correlation/` |
| Run data assimilation | `paper_scripts/run_data_assimilation.py` | `config_files/data_assimilation/` |

For example:

```
python paper_scripts/run_data_assimilation.py --config config_files/data_assimilation/lorenz96_baselineinit_obsopt.config
```

The configs read and write under `/data/invobs-da-data/`. To skip training, download the authors' trained models and correlation files with [gsutil](https://cloud.google.com/storage/docs/gsutil):

```
mkdir -p /data
gsutil cp -r gs://gresearch/jax-cfd/projects/invobs-data-assimilation/invobs-da-data /data
```

The original authors report about 1.5 hours for Lorenz96 and about 10 hours for Kolmogorov flow on one V100 GPU for a data assimilation run.

`config_files/test/` holds smaller configs for the tests in `paper_scripts/tests/`.

## Module map

- `dynamical_system.py` defines `DynamicalSystem`, `Lorenz96`, and `KolmogorovFlow`. `generate_dyn_sys(config)` picks one from `config['dyn_sys']`.
- `da_methods.py` defines `da_loss_fn`, one parameterized loss that serves both spaces, and `optimize_lbfgs_scipy`.
- `run_data_assimilation.py` builds the observation-space and physics-space losses in `generate_loss_functions` and runs them in order in `optimize_da`. The config fields `physics_space_opt_steps` and `obs_space_opt_steps` set how many L-BFGS steps each stage gets. A run with both is the hybrid method.
- `lorenz96_ml.py` and `kolmogorov_ml.py` define the inverse observation networks `ObservationInverterLorenz96` and `ObservationInverterKolmogorov`. `ml_methods.py` holds `create_model`, `load_model`, and `create_adam_optimizer`.
- `lorenz96_methods.py` and `kolmogorov_methods.py` hold system-specific data generation and baseline initializations.
- `analysis_util.py` holds the plotting helpers for `Analysis_Lorenz96.ipynb` and `Analysis_KolmogorovFlow.ipynb`.

`run_data_assimilation.py` sets `XLA_FLAGS` and `XLA_PYTHON_CLIENT_PREALLOCATE` at import time for reproducibility and to avoid out-of-memory errors. Keep these settings if you change the entry point.

## Correlation preconditioning

`generate_correlation_transform` loads the stored square root and inverse square root of the state covariance. It returns one function, `correlation_transform(x, mode)`, where `mode` is `'cor'` or `'dec'`. L-BFGS optimizes in decorrelated coordinates, and the result is correlated back to physical coordinates at the end.
