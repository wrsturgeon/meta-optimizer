import distributions, feedforward, permutations

from beartype import beartype
from hypothesis import given, settings, strategies as st, Verbosity
from hypothesis.extra import numpy as hnp
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax.numpy import linalg as jla
from jaxtyping import jaxtyped, Array, Float, TypeCheckError
from math import prod
from numpy.typing import ArrayLike
from os import environ
import pytest


TEST_COUNT_CI = 10000
TEST_COUNT_NORMAL = 100
settings.register_profile(
    "no_deadline",
    deadline=None,
    derandomize=True,
    max_examples=TEST_COUNT_NORMAL,
)
settings.register_profile(
    "ci",
    parent=settings.get_profile("no_deadline"),
    max_examples=TEST_COUNT_CI,
    verbosity=Verbosity.verbose,
)


gh_ci = environ.get("GITHUB_CI")
if gh_ci == "1":
    print("***** Running in CI mode")
    settings.load_profile("ci")
    TEST_COUNT = TEST_COUNT_CI
else:
    print(
        f'***** NOT running in CI mode (environment variable `GITHUB_CI` is `{gh_ci}`, which is not `"1"`)'
    )
    settings.load_profile("no_deadline")
    TEST_COUNT = TEST_COUNT_NORMAL


@jaxtyped(typechecker=beartype)
def prop_normalize_no_axis(x: Float[Array, "..."]):
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
@jaxtyped(typechecker=beartype)
def test_normalize_no_axis_prop(x: ArrayLike):
    prop_normalize_no_axis(jnp.array(x))


# Identical to the above, except along one specific axis
@jaxtyped(typechecker=beartype)
def prop_normalize_with_axis(x: Float[Array, "..."], axis: int):
    assert 0 <= axis < x.ndim
    if not (jnp.isfinite(jnp.mean(x)) and jnp.isfinite(jnp.std(x))):
        return
    auto = distributions.normalize(x, axis)
    manual = jnp.apply_along_axis(distributions.normalize, axis, x)
    zero_variance = jnp.all(
        jnp.isclose(jnp.apply_along_axis(lambda z: z[0:1], axis, x), x),
        axis=axis,
        keepdims=True,
    )
    loss = jnp.abs(auto - manual)
    good = jnp.logical_or(zero_variance, loss < 0.01)
    assert jnp.all(good)


@jaxtyped(typechecker=beartype)
def test_normalize_with_axis_1():
    prop_normalize_with_axis(jnp.zeros([3, 3, 3]), axis=1)


@jaxtyped(typechecker=beartype)
def test_normalize_with_axis_2():
    prop_normalize_with_axis(jnp.ones([3, 3, 3]), axis=0)


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
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
@jaxtyped(typechecker=beartype)
def test_normalize_with_axis_prop(x: ArrayLike, axis: int):
    prop_normalize_with_axis(jnp.array(x), axis=axis)


@jaxtyped(typechecker=beartype)
def prop_kabsch(to_be_rotated: Float[Array, "..."], target: Float[Array, "..."]):
    assert to_be_rotated.shape == target.shape
    norm1 = jla.norm(to_be_rotated)
    norm2 = jla.norm(target)
    if (
        jnp.allclose(0, to_be_rotated)
        or jnp.allclose(0, target)
        or not (
            jnp.all(jnp.isfinite(to_be_rotated))
            and jnp.all(jnp.isfinite(target))
            and jnp.isfinite(norm1)
            and jnp.isfinite(norm2)
        )
    ):
        return
    to_be_rotated /= norm1
    target /= norm2
    R = distributions.kabsch(to_be_rotated, target)
    rotated = to_be_rotated @ R
    loss = jnp.abs(rotated - target)
    assert jnp.all(loss < 0.01)


@jaxtyped(typechecker=beartype)
def test_kabsch_1():
    eye = jnp.eye(1, 4).reshape(1, 1, 4)
    actual = distributions.kabsch(eye, eye)
    assert jnp.allclose(actual, jnp.eye(4, 4).reshape(1, 4, 4))


@jaxtyped(typechecker=beartype)
def test_kabsch_2():
    prop_kabsch(
        jnp.array([[[1, 0, 0, 0]]], dtype=jnp.float32),
        jnp.array([[[0, 1, 0, 0]]], dtype=jnp.float32),
    )


@jaxtyped(typechecker=beartype)
def test_kabsch_3():
    prop_kabsch(jnp.ones([1, 1, 4]), jnp.ones([1, 1, 4]))


@jaxtyped(typechecker=beartype)
def test_kabsch_4():
    prop_kabsch(jnp.ones([1, 1, 4]), jnp.array([[[0, 1, 1, 1]]], dtype=jnp.float32))


@jaxtyped(typechecker=beartype)
def test_kabsch_5():
    prop_kabsch(
        to_be_rotated=jnp.array([[[1, 1, 1, 1]]], dtype=jnp.float32),
        target=jnp.array(
            [[[9.223372e18, 9.223372e18, 9.223372e18, 9.223372e18]]], dtype=jnp.float32
        ),
    )


@given(
    hnp.arrays(dtype=jnp.float32, shape=[1, 1, 4]),
    hnp.arrays(dtype=jnp.float32, shape=[1, 1, 4]),
)
@jaxtyped(typechecker=beartype)
def test_kabsch_prop(to_be_rotated: ArrayLike, target: ArrayLike):
    prop_kabsch(jnp.array(to_be_rotated), jnp.array(target))


@given(hnp.arrays(dtype=jnp.float32, shape=[3, 3]))
@jaxtyped(typechecker=beartype)
def test_feedforward_id_prop(np_x: ArrayLike):
    x = jnp.array(np_x)
    if not jnp.all(jnp.isfinite(x)):
        return
    y = feedforward.feedforward([jnp.eye(3, 3)], [jnp.zeros([3, 3])], x, nl=lambda z: z)
    assert jnp.allclose(y, x)


# NOTE: The big problem with using rotation matrices is that,
# with practically all nonlinearities (e.g. ReLU or GELU),
# negative values are effectively eliminated whereas
# positive values are allowed to pass effectively unchanged.
# Rotation matrices use negative values extensively, but
# negation significantly (very significantly!) alters behavior.


@jaxtyped(typechecker=beartype)
def test_feedforward_init_1():
    W, B = feedforward.feedforward_init([5, 42, 3, 8, 7], jrnd.PRNGKey(42))
    y = feedforward.feedforward(W, B, jnp.ones([5]))
    assert jnp.allclose(
        y,
        jnp.array(
            [
                0.40948272,
                0.14077032,
                -0.05583315,
                -0.15906703,
                -0.05009099,
                0.2515578,
                -0.13697886,
            ]
        ),
    )


@jaxtyped(typechecker=beartype)
def prop_rotating_weights(
    W: list[Float[Array, "..."]],
    B: list[Float[Array, "..."]],
    x: Array,
    angles: list[Float[Array, "..."]],
):
    assert len(angles) == len(W)
    if not all([jnp.all(jnp.isfinite(angle)) for angle in angles]):
        return
    angle_array = jnp.array(angles)[:, jnp.newaxis]
    R = list(distributions.kabsch(angle_array[:-1], angle_array[1:]))  # batched!
    WR, BR = feedforward.rotate_weights(W, B, R)
    y = feedforward.feedforward(W, B, x, nl=lambda z: z)
    yR = feedforward.feedforward(WR, BR, x, nl=lambda z: z)
    assert jnp.allclose(y, yR)


@jaxtyped(typechecker=beartype)
def test_rotating_weights_1():
    prop_rotating_weights(
        [jnp.eye(3, 3)],
        [jnp.eye(1, 3)[0]],
        jnp.eye(1, 3)[0],
        [jnp.array([1, 0, 0], dtype=jnp.float32)],
    )


@jaxtyped(typechecker=beartype)
def test_rotating_weights_2():
    prop_rotating_weights(
        [jnp.eye(3, 3), jnp.eye(3, 3)],
        [jnp.eye(1, 3)[0], jnp.eye(1, 3)[0]],
        jnp.eye(1, 3)[0],
        [
            jnp.array([1, 0, 0], dtype=jnp.float32),
            jnp.array([0, 1, 0], dtype=jnp.float32),
        ],
    )


@jaxtyped(typechecker=beartype)
def test_rotating_weights_3():
    prop_rotating_weights(
        W=list(jnp.ones([2, 3, 3])),
        B=list(jnp.zeros([2, 3])),
        x=jnp.ones([3]),
        angles=[
            jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32),
            jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
        ],
    )


@jaxtyped(typechecker=beartype)
def test_rotating_weights_prop_fake():
    for seed in range(TEST_COUNT):
        k1, k2, k3 = jrnd.split(jrnd.PRNGKey(seed), 3)
        W = list(jnn.initializers.he_normal()(k1, [2, 3, 3]))
        B = list(jnp.zeros([2, 3]))
        x = jrnd.normal(k2, [3])
        angles = list(jrnd.normal(k3, [2, 3]))
        prop_rotating_weights(W, B, x, angles)


# NOTE: THIS TAKES OVER TEN HOURS; use the above (identical but w/o shrinking) instead
# @given(
#     hnp.arrays(dtype=jnp.float32, shape=[layers, ndim, ndim]),
#     hnp.arrays(dtype=jnp.float32, shape=[layers, ndim]),
#     hnp.arrays(dtype=jnp.float32, shape=[ndim]),
#     hnp.arrays(dtype=jnp.float32, shape=[layers, ndim]),
# )
# def test_rotating_weights_prop_1_layer(
#     W: Array,
#     B: Array,
#     x: Array,
#     angles: Array,
# ):
#     prop_rotating_weights(list(W), list(B), x, list(angles))


# TODO: test varying dimensionality across layers


@jaxtyped(typechecker=beartype)
def prop_permutation_conjecture(r1: Float[Array, "n"], r2: Float[Array, "n"]):
    # these lines (this comment +/- 1) are effectively just
    # generating a random rotation matrix stored in `R`
    R = distributions.kabsch(r1, r2)
    assert R.shape[0] == 1
    R = R[0]

    # now that we have a rotation matrix,
    # can we obtain the closest permutation matrix
    # by simply taking the maximum along each row/column?
    # test whether row/column methods match, which would imply the above:
    Rabs = jnp.abs(R)
    outputs = []
    for axis in range(R.ndim):
        outputs.append(
            jnn.one_hot(jnp.argmax(Rabs, axis=axis), R.shape[axis], axis=axis)
        )
        assert outputs[-1].shape == R.shape
    for output in outputs:
        for cmp in outputs:
            assert jnp.allclose(output, cmp)


# Turns out the above does not hold:
@jaxtyped(typechecker=beartype)
def test_permutation_conjecture_disproven():
    r1 = jnp.array([[[0.21653588, -0.6419788, 1.1067219]]], dtype=jnp.float32)
    r2 = jnp.array([[[-0.9220991, 2.6091485, -1.3119074]]], dtype=jnp.float32)
    try:
        prop_permutation_conjecture(r1, r2)
        raise
    except:
        return


# TODO: Write a 2^n-time ( :( ) checker for the above,
# keeping track of a running best-so-far and its score,
# then see how long it actually takes.


@jaxtyped(typechecker=beartype)
def test_permute_1():
    x = jnp.array(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ],
        dtype=jnp.float32,
    )
    i = jnp.array([3, 1, 4, 2, 0], dtype=jnp.uint)
    y = permutations.permute(x, i)
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [0, 0, 0, 4, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 5],
                [0, 0, 3, 0, 0],
                [1, 0, 0, 0, 0],
            ]
        ),
    )


def test_permute_typechecking():
    x = jnp.eye(5, 5, dtype=jnp.float32)
    i = jnp.array(range(6), dtype=jnp.uint)
    with pytest.raises(TypeCheckError):
        permutations.permute(x, i)
