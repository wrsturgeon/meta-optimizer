import distributions

from hypothesis import given, settings, strategies as st, Verbosity
from hypothesis.extra import numpy as hnp
from jax import jit, numpy as jnp
from os import environ


settings.register_profile(
    "no_deadline",
    deadline=None,
    derandomize=True,
    max_examples=1000,
)
settings.register_profile(
    "ci",
    parent=settings.get_profile("no_deadline"),
    max_examples=100000,
    verbosity=Verbosity.verbose,
)


if environ.get("GITHUB_CI") == "1":
    print("***** Running in CI mode")
    settings.load_profile("ci")
else:
    print("***** NOT running in CI mode")
    settings.load_profile("no_deadline")


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
