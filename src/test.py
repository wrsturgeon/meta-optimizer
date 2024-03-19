import distributions

from hypothesis import given, settings, strategies as st, Verbosity
from hypothesis.extra import numpy as hnp
from jax import Array, jit, numpy as jnp
from jax.numpy import linalg as jla
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


def prop_normalize_no_axis(x: Array):
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
    prop_normalize_no_axis(x)


# Identical to the above, except along one specific axis
def prop_normalize_with_axis(x: Array, axis: int):
    assert 0 <= axis < x.ndim
    if not (jnp.isfinite(jnp.mean(x)) and jnp.isfinite(jnp.std(x))):
        return
    auto = distributions.normalize(x, axis)
    manual = jnp.apply_along_axis(distributions.normalize, axis, x)
    print(f"LHZLKJHDGLKSH: {jnp.apply_along_axis(lambda z: z[0:1], axis, x)}")
    zero_variance = jnp.all(
        jnp.isclose(jnp.apply_along_axis(lambda z: z[0:1], axis, x), x),
        axis=axis,
        keepdims=True,
    )
    print(f"zero_variance = {zero_variance}")
    loss = jnp.abs(auto - manual)
    print(f"loss = {loss}")
    good = jnp.logical_or(zero_variance, loss < 0.01)
    print(f"good = {good}")
    assert jnp.all(good)


def test_normalize_with_axis_1():
    prop_normalize_with_axis(jnp.zeros([3, 3, 3]), axis=1)


def test_normalize_with_axis_2():
    prop_normalize_with_axis(jnp.ones([3, 3, 3]), axis=0)


def test_normalize_with_axis_3():
    prop_normalize_with_axis(
        x=jnp.array(
            [
                [
                    [jnp.nan, 0.0000000e00, 1.8206815e16],
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                ],
                [
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                ],
                [
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                    [1.8297095e16, 1.8297095e16, 1.8297095e16],
                ],
            ],
            dtype=jnp.float32,
        ),
        axis=0,
    )


def test_normalize_with_axis_4():
    prop_normalize_with_axis(
        x=jnp.array(
            [
                [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=jnp.float32,
        ),
        axis=1,
    )


@given(hnp.arrays(dtype=jnp.float32, shape=[3, 3, 3]), st.integers(0, 2))
def test_prop_normalize_with_axis(x: Array, axis: int):
    prop_normalize_with_axis(x, axis=axis)


def prop_kabsch(to_be_rotated: Array, target: Array):
    if (
        jnp.allclose(0, to_be_rotated)
        or jnp.allclose(0, target)
        or not (jnp.all(jnp.isfinite(to_be_rotated)) and jnp.all(jnp.isfinite(target)))
    ):
        return
    to_be_rotated /= jla.norm(to_be_rotated)
    target /= jla.norm(target)
    R = distributions.kabsch(to_be_rotated, target)
    rotated = to_be_rotated @ R
    loss = jnp.abs(rotated - target)
    assert jnp.all(loss < 0.01)


def test_kabsch_1():
    eye = jnp.eye(1, 4).reshape(1, 1, 4)
    actual = distributions.kabsch(eye, eye)
    assert jnp.allclose(actual, jnp.eye(4, 4).reshape(1, 4, 4))


def test_kabsch_2():
    prop_kabsch(jnp.array([[[1, 0, 0, 0]]]), jnp.array([[[0, 1, 0, 0]]]))


def test_kabsch_3():
    prop_kabsch(jnp.ones([1, 1, 4]), jnp.ones([1, 1, 4]))


def test_kabsch_4():
    prop_kabsch(jnp.ones([1, 1, 4]), jnp.array([[[0, 1, 1, 1]]]))


@given(
    hnp.arrays(dtype=jnp.float32, shape=[1, 1, 4]),
    hnp.arrays(dtype=jnp.float32, shape=[1, 1, 4]),
)
def test_prop_kabsch(to_be_rotated: Array, target: Array):
    prop_kabsch(to_be_rotated, target)
