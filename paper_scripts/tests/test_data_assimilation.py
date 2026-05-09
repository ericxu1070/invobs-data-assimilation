# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Unit tests for methods in run_data_assimilation.py using pytest.

This test depends on the pinned legacy JAX stack (jax==0.2.6, flax==0.3.0,
jax_cfd==0.1.0) and is skipped automatically when those modules are absent --
see CLAUDE.md for the reason the stack is not installed by default.

Note: the original file imported ``from data_assimilation import ...`` but the
module on disk is ``run_data_assimilation.py``; that mismatch was flagged in
INSTRUCTIONS_FROM_REVIEW.md (Phase-cross-cutting task). The import below uses
the actual module name. Fix is import-only -- module was NOT renamed.
"""
import pytest
import json
import numpy as np

# Skip this whole file unless the legacy JAX stack is importable.
jax = pytest.importorskip('jax', reason='requires legacy jax==0.2.6')
jnp = pytest.importorskip('jax.numpy', reason='requires legacy jax.numpy==0.2.6')
pytest.importorskip('flax', reason='requires legacy flax==0.3.0')
pytest.importorskip('jax_cfd', reason='requires legacy jax_cfd==0.1.0')

from run_data_assimilation import generate_correlation_transform  # noqa: E402

SEED = 7
ATOL = 1e-3


@pytest.mark.parametrize(
    'config_filename',
    [
        'config_files/test/lorenz96_test.config',
        'config_files/test/kolmogorov_test.config',
    ],
)
def test_correlation_transform(config_filename):
    with open(config_filename, 'r') as config_file:
      config = json.load(config_file)
    correlation_transform = generate_correlation_transform(config)
    if config['dyn_sys'] == 'kolmogorov':
      x_shape = (64, 64, 2)
    elif config['dyn_sys'] == 'lorenz96':
      x_shape = (40,)
    else:
      raise ValueError('Dynamical system not implemented.')
    prng_key = jax.random.PRNGKey(SEED)
    x = jax.random.normal(prng_key, x_shape)
    np.testing.assert_allclose(
        correlation_transform(correlation_transform(x, 'cor'), 'dec'),
        x,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        correlation_transform(correlation_transform(x, 'dec'), 'cor'),
        x,
        atol=ATOL,
    )
  