import distributions

from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
from jax import jit, numpy as jnp


@settings(deadline=None)
@given(hnp.arrays(dtype=jnp.float32, shape=[3, 3, 3]))
def test_normalize_no_axis(x):
    print()
    if x.size == 0 or not jnp.all(jnp.isfinite(x)):
        return
    # Standard deviation is subject to (significant!) numerical error, so
    # we have to check if all elements are literally equal....
    all_same = jnp.all(jnp.isclose(x.ravel()[0], x))
    if (not all_same) and jnp.isfinite(jnp.std(x)):
        print(f"Mean: {jnp.mean(x)}; Standard Deviation: {jnp.std(x)}")
        x = distributions.normalize(x)
        print(f"Mean: {jnp.mean(x)}; Standard Deviation: {jnp.std(x)}")
        assert jnp.abs(jnp.mean(x)) < 0.01
        assert jnp.abs(jnp.std(x) - 1) < 0.01
