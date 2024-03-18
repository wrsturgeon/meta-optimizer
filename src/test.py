import distributions

from hypothesis import given, settings, strategies as st, Verbosity
from hypothesis.extra import numpy as hnp
from jax import Array, jit, numpy as jnp
from math import prod
from os import environ


settings.register_profile(
    "no_deadline",
    deadline=None,
    derandomize=True,
    max_examples=100,
)
settings.register_profile(
    "ci",
    parent=settings.get_profile("no_deadline"),
    max_examples=100000,
    verbosity=Verbosity.verbose,
)


gh_ci = environ.get("GITHUB_CI")
if gh_ci == "1":
    print("***** Running in CI mode")
    settings.load_profile("ci")
else:
    print(
        f'***** NOT running in CI mode (environment variable `GITHUB_CI` is `{gh_ci}`, which is not `"1"`)'
    )
    settings.load_profile("no_deadline")


def normalize_no_axis(x: Array):
    if x.size == 0 or not jnp.all(jnp.isfinite(x)):
        return
    # Standard deviation is subject to (significant!) numerical error, so
    # we have to check if all elements are literally equal.
    zero_variance = jnp.all(jnp.isclose(x.ravel()[0], x))
    if (not zero_variance) and jnp.isfinite(jnp.std(x)):
        y = distributions.normalize(x)
        assert jnp.abs(jnp.mean(y)) < 0.01
        assert jnp.abs(jnp.std(y) - 1) < 0.01


@given(hnp.arrays(dtype=jnp.float32, shape=[3, 3, 3]))
def test_prop_normalize_no_axis(x: Array):
    normalize_no_axis(x)


# Identical to the above, except along one specific axis
def normalize_with_axis(x: Array, axis: int):
    assert 0 <= axis < x.ndim
    auto = distributions.normalize(x, axis)
    manual = jnp.apply_along_axis(distributions.normalize, axis, x)
    print()
    print("Automatic:")
    print(auto)
    print()
    print("Manual:")
    print(manual)
    print()
    assert jnp.all(
        jnp.logical_or(
            jnp.logical_or(
                # Either both are infinite, ...
                jnp.logical_not(
                    jnp.logical_or(jnp.isfinite(auto), jnp.isfinite(manual))
                ),
                # ...the original has zero variance, ...
                jnp.all(
                    jnp.isclose(jnp.apply_along_axis(lambda z: z[0], axis, x), x),
                    axis=axis,
                ),
            ),
            # ...or both are identical
            jnp.isclose(auto, manual),
        )
    )


def test_normalize_with_axis_1():
    normalize_with_axis(jnp.zeros([3, 3, 3]), axis=1)


def test_normalize_with_axis_2():
    normalize_with_axis(jnp.ones([3, 3, 3]), axis=0)


# @given(hnp.arrays(dtype=jnp.float32, shape=[3, 3, 3]), st.integers(0, 2))
# def test_prop_normalize_with_axis(x: Array, axis: int):
#     normalize_with_axis(x, axis=axis)
