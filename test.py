from metaoptimizer import distributions, feedforward, permutations, stock_optimizers
from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Any, Callable, Protocol
from functools import partial
from hypothesis import given, settings, strategies as st, Verbosity
from hypothesis.extra import numpy as hnp
from jax import jit, grad, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental import checkify
from jax.numpy import linalg as jla
from jaxtyping import jaxtyped, Array, Float, TypeCheckError, UInt
from math import prod
from numpy.typing import ArrayLike
from os import environ
import pytest


TEST_COUNT_CI = 1000
TEST_COUNT_NORMAL = 10
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
    # verbosity=Verbosity.verbose,
)


gh_ci = environ.get("GITHUB_CI")
if gh_ci == "1":  # pragma: no cover
    print("***** Running in CI mode")
    settings.load_profile("ci")
    TEST_COUNT = TEST_COUNT_CI
else:  # pragma: no cover
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


@given(hnp.arrays(dtype=jnp.float32, shape=(3, 3, 3)))
@jaxtyped(typechecker=beartype)
def test_normalize_no_axis_prop(x: ArrayLike):
    prop_normalize_no_axis(jnp.array(x))


@jaxtyped(typechecker=beartype)
def test_normalize_no_axis():
    prop_normalize_no_axis(jnp.array([-1, 1, -1, 1, -1], dtype=jnp.float32))


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


@given(hnp.arrays(dtype=jnp.float32, shape=(3, 3, 3)), st.integers(0, 2))
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
    hnp.arrays(dtype=jnp.float32, shape=(1, 1, 4)),
    hnp.arrays(dtype=jnp.float32, shape=(1, 1, 4)),
)
@jaxtyped(typechecker=beartype)
def test_kabsch_prop(to_be_rotated: ArrayLike, target: ArrayLike):
    prop_kabsch(jnp.array(to_be_rotated), jnp.array(target))


@jaxtyped(typechecker=beartype)
def test_rotate_and_compare_1():
    ideal = jnp.eye(5, 5)[jnp.newaxis]
    actual = jnp.array(
        [
            [
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        ],
        dtype=jnp.float32,
    )
    norm, rotated, R = distributions.rotate_and_compare(actual, ideal)
    assert jnp.allclose(norm, 0)
    assert jnp.allclose(rotated, ideal)
    assert jnp.allclose(R, actual.transpose(0, 2, 1))


@given(hnp.arrays(dtype=jnp.float32, shape=(3, 3)))
@jaxtyped(typechecker=beartype)
def test_feedforward_id_prop(np_x: ArrayLike):
    x = jnp.array(np_x)
    if not jnp.all(jnp.isfinite(x)):
        return
    err, y = feedforward.feedforward(
        Weights([jnp.eye(3, 3)], [jnp.zeros([3])]), x, nl=lambda z: z
    )
    err.throw()
    assert jnp.allclose(y, x)


# NOTE: The big problem with using rotation matrices is that,
# with practically all nonlinearities (e.g. ReLU or GELU),
# negative values are effectively eliminated whereas
# positive values are allowed to pass effectively unchanged.
# Rotation matrices use negative values extensively, but
# negation significantly (very significantly!) alters behavior.


@jaxtyped(typechecker=beartype)
def test_feedforward_init_1():
    w = feedforward.feedforward_init([5, 42, 3, 8, 7], jrnd.PRNGKey(42))
    err, y = feedforward.feedforward(w, jnp.ones([1, 5]), nl=jnn.gelu)
    err.throw()
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
    w: Weights,
    x: Float[Array, "batch ndim"],
    angles: list[Float[Array, "..."]],
):
    assert len(angles) == w.layers()
    if not all([jnp.all(jnp.isfinite(angle)) for angle in angles]):
        return
    angle_array = jnp.array(angles)[:, jnp.newaxis]
    R = list(distributions.kabsch(angle_array[:-1], angle_array[1:]))  # batched!
    wR = feedforward.rotate_weights(w, R)
    err, y = feedforward.feedforward(w, x, nl=lambda z: z)
    err.throw()
    err, yR = feedforward.feedforward(wR, x, nl=lambda z: z)
    err.throw()
    assert jnp.allclose(y, yR)


@jaxtyped(typechecker=beartype)
def test_rotating_weights_1():
    prop_rotating_weights(
        Weights([jnp.eye(3, 3)], [jnp.eye(1, 3)[0]]),
        jnp.eye(3, 3),
        [jnp.array([1, 0, 0], dtype=jnp.float32)],
    )


@jaxtyped(typechecker=beartype)
def test_rotating_weights_2():
    prop_rotating_weights(
        Weights([jnp.eye(3, 3), jnp.eye(3, 3)], [jnp.eye(1, 3)[0], jnp.eye(1, 3)[0]]),
        jnp.eye(3, 3),
        [
            jnp.array([1, 0, 0], dtype=jnp.float32),
            jnp.array([0, 1, 0], dtype=jnp.float32),
        ],
    )


@jaxtyped(typechecker=beartype)
def test_rotating_weights_3():
    prop_rotating_weights(
        w=Weights(list(jnp.ones([2, 3, 3])), list(jnp.zeros([2, 3]))),
        x=jnp.ones([1, 3]),
        angles=[
            jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32),
            jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
        ],
    )


@jaxtyped(typechecker=beartype)
def test_rotating_weights_4():
    prop_rotating_weights(
        w=Weights([jnp.zeros([0, 0, 0])], [jnp.zeros([0, 0])]),
        x=jnp.array([[]]),
        angles=[jnp.array([jnp.inf])],
    )


@jaxtyped(typechecker=beartype)
def test_rotating_weights_prop_fake():
    for seed in range(TEST_COUNT):
        k1, k2, k3 = jrnd.split(jrnd.PRNGKey(seed), 3)
        W = list(jnn.initializers.he_normal()(k1, [2, 3, 3]))
        B = list(jnp.zeros([2, 3]))
        x = jrnd.normal(k2, [3, 3])
        angles = list(jrnd.normal(k3, [2, 3]))
        prop_rotating_weights(Weights(W, B), x, angles)


# NOTE: THIS TAKES OVER TEN HOURS; use the above (identical but w/o shrinking) instead
# @given(
#     hnp.arrays(dtype=jnp.float32, shape=(layers, ndim, ndim)),
#     hnp.arrays(dtype=jnp.float32, shape=(layers, ndim)),
#     hnp.arrays(dtype=jnp.float32, shape=(ndim)),
#     hnp.arrays(dtype=jnp.float32, shape=(layers, ndim)),
# )
# def test_rotating_weights_prop_1_layer(
#     W: Array,
#     B: Array,
#     x: Array,
#     angles: Array,
# ):
#     prop_rotating_weights(Weights(list(W), list(B)), x, list(angles))


# TODO: test varying dimensionality across layers


@jaxtyped(typechecker=beartype)
def prop_permutation_conjecture(
    r1: Float[Array, "batch points ndim"],
    r2: Float[Array, "batch points ndim"],
):
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
    with pytest.raises(AssertionError):
        prop_permutation_conjecture(r1, r2)


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
    i = jnp.array([3, 1, 4, 2, 0], dtype=jnp.uint32)
    y = permutations.permute(x, i, axis=0)
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


@jaxtyped(typechecker=beartype)
def test_permute_2():
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
    i = jnp.array([3, 1, 4, 2, 0], dtype=jnp.uint32)
    y = permutations.permute(x, i, axis=1)
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [0, 0, 0, 0, 1],
                [0, 2, 0, 0, 0],
                [0, 0, 0, 3, 0],
                [4, 0, 0, 0, 0],
                [0, 0, 5, 0, 0],
            ]
        ),
    )


@jaxtyped(typechecker=beartype)
def test_permute_size_check():
    x = jnp.eye(5, 5, dtype=jnp.float32)
    i = jnp.array(range(6), dtype=jnp.uint32)
    with pytest.raises(AssertionError):
        permutations.permute(x, i, axis=0)


@jaxtyped(typechecker=beartype)
def test_permute_axis_check():
    x = jnp.eye(5, 5, dtype=jnp.float32)
    i = jnp.array(range(5), dtype=jnp.uint32)
    with pytest.raises(IndexError):
        permutations.permute(x, i, axis=5)


@jaxtyped(typechecker=beartype)
def test_find_permutation_1():
    Wideal = jnp.eye(5, 5, dtype=jnp.float32)
    Bideal = jnp.ones(5, dtype=jnp.float32)
    Wactual = jnp.array(
        [
            [0, 0, -1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, -1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    Bactual = jnp.array([-1, 1, -1, 1, 1], dtype=jnp.float32)
    p = permutations.find_permutation(
        Wactual, Bactual, Wideal, Bideal, jnp.array(range(5), dtype=jnp.uint32)
    )
    assert jnp.all(p.indices == jnp.array([2, 0, 3, 1, 4], dtype=jnp.uint32))
    assert jnp.all(
        p.flip == jnp.array([True, False, True, False, False], dtype=jnp.bool)
    )
    assert jnp.allclose(p.loss, 0)


@jaxtyped(typechecker=beartype)
def prop_permute_hidden_layers(
    w: Weights,
    p: list[UInt[Array, "..."]],
    x: Float[Array, "batch ndim"],
):
    assert w.layers() == len(p) + 1
    if (
        (not all([jnp.all(jnp.isfinite(w)) for w in w.W]))
        or (not all([jnp.all(jnp.isfinite(b)) for b in w.B]))
        or (not jnp.all(jnp.isfinite(x)))
        or any([jnp.any(jnp.sum(jnp.square(w)) > 1e5) for w in w.W])
        or any([jnp.any(jnp.sum(jnp.square(b)) > 1e5) for b in w.B])
        or jnp.sum(jnp.square(x)) > 1e5
    ):
        return
    wp = permutations.permute_hidden_layers(w, p)
    err, y = feedforward.feedforward(w, x, nl=jnn.gelu)
    err.throw()
    err, yp = feedforward.feedforward(wp, x, nl=jnn.gelu)
    err.throw()
    assert jnp.allclose(y, yp)


@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_1():
    prop_permute_hidden_layers(
        Weights(
            [jnp.zeros([5, 5]) for _ in range(3)], [jnp.zeros([5]) for _ in range(3)]
        ),
        [jnp.array(range(5), dtype=jnp.uint32), jnp.array(range(5), dtype=jnp.uint32)],
        jnp.zeros([1, 5]),
    )


@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_2():
    prop_permute_hidden_layers(
        Weights(
            list(jnp.full([3, 5, 5], 3.2994486e17)),
            list(jnp.full([3, 5], 5.442048e17)),
        ),
        p=[jnp.array(range(5), dtype=jnp.uint32) for _ in range(2)],
        x=jnp.full([1, 5], 1.3602583e17),
    )


@given(
    st.lists(hnp.arrays(dtype=jnp.float32, shape=(5, 5)), min_size=3, max_size=3),
    st.lists(hnp.arrays(dtype=jnp.float32, shape=(5)), min_size=3, max_size=3),
    st.lists(st.permutations(range(5)), min_size=2, max_size=2),
    hnp.arrays(dtype=jnp.float32, shape=(1, 5)),
)
@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_prop(
    W: list[ArrayLike],
    B: list[ArrayLike],
    P: list[list[int]],
    x: ArrayLike,
):
    prop_permute_hidden_layers(
        Weights([jnp.array(w) for w in W], [jnp.array(b) for b in B]),
        [jnp.array(p, dtype=jnp.uint32) for p in P],
        jnp.array(x),
    )


@jaxtyped(typechecker=beartype)
def test_layer_distance():
    eye = jnp.eye(5, 5)
    perm = jnp.array([3, 1, 4, 2, 0], dtype=jnp.uint32)
    inv_perm = jnp.array([4, 1, 3, 0, 2], dtype=jnp.uint32)
    Wactual = [
        permutations.permute(eye, perm, axis=0),
        permutations.permute(eye, inv_perm, axis=0),
        eye,
    ]
    Bactual = [jnp.zeros(5) for _ in range(3)]
    Wideal = [eye, eye, eye]
    Bideal = [jnp.zeros(5) for _ in range(3)]
    last_best = [jnp.array(range(5), dtype=jnp.uint32) for _ in range(2)]
    loss, ps = permutations.layer_distance(
        Weights(Wactual, Bactual),
        Weights(Wideal, Bideal),
        last_best,
    )
    assert len(ps) == 2
    assert jnp.isclose(loss, 0)
    assert jnp.all(ps[0].indices == perm)
    assert jnp.all(jnp.logical_not(ps[0].flip))
    assert jnp.all(ps[1].indices == inv_perm)
    assert jnp.all(jnp.logical_not(ps[1].flip))


NDIM = 5
LAYERS = 3


@jaxtyped(typechecker=beartype)
def prop_optim(optim: Callable[[Weights, Weights], tuple[Any, Weights]]):

    @jaxtyped(typechecker=beartype)
    def raw_loss(w: Weights, x: Float[Array, "batch ndim"]) -> Float[Array, ""]:
        err, y = feedforward.feedforward(w, x, nl=jnn.gelu)
        err.throw()
        return jnp.sum(jnp.abs(jnp.sin(x) - y))

    loss = jit(checkify.checkify(raw_loss))
    dLdw = jit(checkify.checkify(grad(raw_loss, argnums=0)))
    w = feedforward.feedforward_init(
        [NDIM for _ in range(LAYERS + 1)], jrnd.PRNGKey(42)
    )
    key = jrnd.PRNGKey(42)
    orig_x = jrnd.uniform(key, [1, NDIM])
    err, orig_loss = loss(w, orig_x)
    err.throw()
    for _ in range(100):
        k, key = jrnd.split(key)
        err, d = dLdw(w, jrnd.uniform(k, [1, NDIM]))
        err.throw()
        optim, w = optim(w, d)
    err, post_loss = loss(w, orig_x)
    err.throw()
    # make sure we learned *something*:
    assert post_loss < orig_loss


@jaxtyped(typechecker=beartype)
def test_optim_sgd():
    prop_optim(stock_optimizers.SGD(0.01))


@jaxtyped(typechecker=beartype)
def test_optim_momentum():
    prop_optim(
        stock_optimizers.Momentum(
            lr=0.01,
            momentum=0.9,
            last_update=Weights(
                [jnp.zeros([NDIM, NDIM]) for _ in range(LAYERS)],
                [jnp.zeros([NDIM]) for _ in range(LAYERS)],
            ),
        )
    )


@jaxtyped(typechecker=beartype)
def test_optim_nesterov():
    prop_optim(
        stock_optimizers.Nesterov(
            lr=0.01,
            momentum=0.9,
            last_update=Weights(
                [jnp.zeros([NDIM, NDIM]) for _ in range(LAYERS)],
                [jnp.zeros([NDIM]) for _ in range(LAYERS)],
            ),
            overstep=0.9,
            actual=Weights(
                [jnp.zeros([NDIM, NDIM]) for _ in range(LAYERS)],
                [jnp.zeros([NDIM]) for _ in range(LAYERS)],
            ),
        )
    )


@jaxtyped(typechecker=beartype)
def test_optim_rmsprop():
    prop_optim(
        stock_optimizers.RMSProp(
            lr=0.01,
            moving_decay=0.9,
            moving_average=Weights(
                [jnp.zeros([NDIM, NDIM]) for _ in range(LAYERS)],
                [jnp.zeros([NDIM]) for _ in range(LAYERS)],
            ),
        )
    )
