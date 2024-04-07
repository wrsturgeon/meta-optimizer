from metaoptimizer import (
    feedforward,
    permutations,
    rotations,
    training,
)
from metaoptimizer.optimizers import Optimizer
from metaoptimizer.training import ForwardPass
from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Any, Callable, Protocol, Tuple
from hypothesis import given, reproduce_failure, settings, strategies as st, Verbosity
from hypothesis.extra import numpy as hnp
from jax import jit, grad, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental.checkify import all_checks, checkify
from jax.numpy import linalg as jla
from jax.tree_util import tree_map, tree_reduce, tree_structure
from jaxtyping import jaxtyped, Array, Bool, Float, PyTree, TypeCheckError, UInt
from math import prod
from numpy.typing import ArrayLike
import operator
from os import environ
import pytest


TEST_COUNT_CI = 10000
TEST_COUNT_NORMAL = 1
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
    R = rotations.kabsch(to_be_rotated, target)
    rotated = to_be_rotated @ R
    loss = jnp.abs(rotated - target)
    assert jnp.all(loss < 0.01)


@jaxtyped(typechecker=beartype)
def test_kabsch_1():
    eye = jnp.eye(1, 4).reshape(1, 1, 4)
    actual = rotations.kabsch(eye, eye)
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
    norm, rotated, R = rotations.rotate_and_compare(actual, ideal)
    assert jnp.allclose(norm, 0)
    assert jnp.allclose(rotated, ideal)
    assert jnp.allclose(R, actual.transpose(0, 2, 1))


@given(hnp.arrays(dtype=jnp.float32, shape=(3, 3)))
@jaxtyped(typechecker=beartype)
def test_feedforward_id_prop(np_x: ArrayLike):
    x = jnp.array(np_x)
    if not jnp.all(jnp.isfinite(x)):
        return
    y = feedforward.run(Weights([jnp.eye(3, 3)], [jnp.zeros([3])]), x, nl=lambda z: z)
    assert jnp.allclose(y, x)


# NOTE: The big problem with using rotation matrices is that,
# with practically all nonlinearities (e.g. ReLU or GELU),
# negative values are effectively eliminated whereas
# positive values are allowed to pass effectively unchanged.
# Rotation matrices use negative values extensively, but
# negation significantly (very significantly!) alters behavior.


@jaxtyped(typechecker=beartype)
def test_feedforward_init_1():
    w = feedforward.init([5, 42, 3, 8, 7], jrnd.PRNGKey(42))
    y = feedforward.run(w, jnp.ones([1, 5]), nl=jnn.gelu)
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


# TODO: test varying dimensionality across layers


@jaxtyped(typechecker=beartype)
def prop_permutation_conjecture(
    r1: Float[Array, "batch points ndim"],
    r2: Float[Array, "batch points ndim"],
):
    # these lines (this comment +/- 1) are effectively just
    # generating a random rotation matrix stored in `R`
    R = rotations.kabsch(r1, r2)
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


def test_permute_3():
    x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.float32)
    i = jnp.array([2, 1, 0], dtype=jnp.uint32)
    assert jnp.allclose(
        permutations.permute(x, i, axis=0),
        jnp.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=jnp.float32),
    )
    assert jnp.allclose(
        permutations.permute(x, i, axis=1),
        jnp.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]], dtype=jnp.float32),
    )


@jaxtyped(typechecker=beartype)
def test_permute_size_check():
    x = jnp.eye(5, 5, dtype=jnp.float32)
    i = jnp.arange(6, dtype=jnp.uint32)
    with pytest.raises(AssertionError):
        permutations.permute(x, i, axis=0)


@jaxtyped(typechecker=beartype)
def test_permute_axis_check():
    x = jnp.eye(5, 5, dtype=jnp.float32)
    i = jnp.arange(5, dtype=jnp.uint32)
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
    p = permutations.find_permutation_weights(Wactual, Bactual, Wideal, Bideal)
    ideal_indices = jnp.array([2, 0, 3, 1, 4], dtype=jnp.uint32)
    ideal_flip = jnp.array([True, False, True, False, False], dtype=jnp.bool)
    assert jnp.all(p.indices == ideal_indices), f"{p.indices} =/= {ideal_indices}"
    assert jnp.all(p.flip == ideal_flip), f"{p.flip} =/= {ideal_flip}"
    assert jnp.allclose(p.loss, 0)


@jaxtyped(typechecker=beartype)
def prop_better_than_random_permutation(
    x: Float[Array, "n m"],
    y: Float[Array, "n m"],
    randomly_chosen_indices: UInt[Array, "n"],
    randomly_chosen_flip: Bool[Array, "n"],
):
    if not (jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y))):
        return
    allegedly_ideal = permutations.find_permutation(x, y)
    print(f"allegedly_ideal = {allegedly_ideal}")
    x = x / (jnp.sqrt(jnp.sum(jnp.square(x), axis=1, keepdims=True)) + 1e-8)
    y = y / (jnp.sqrt(jnp.sum(jnp.square(y), axis=1, keepdims=True)) + 1e-8)
    print("x")
    print(x)
    print("y")
    print(y)
    ap = permutations.permute(y, allegedly_ideal.indices, axis=0)
    rp = permutations.permute(y, randomly_chosen_indices, axis=0)
    print("ap")
    print(ap)
    af = jnp.where(allegedly_ideal.flip[:, jnp.newaxis], -ap, ap)
    rf = jnp.where(randomly_chosen_flip[:, jnp.newaxis], -rp, rp)
    print("af")
    print(af)
    aL = jnp.sum(jnp.abs(af - x))
    rL = jnp.sum(jnp.abs(rf - x))
    assert jnp.isclose(allegedly_ideal.loss, aL), f"{allegedly_ideal.loss} =/= {aL}"
    assert aL <= rL, f"{aL} </= {rL}"


def test_better_than_random_permutation_frozen_1():
    # First, test that we're normalizing the right axes:
    x = jnp.array([[1, 1], [-1, 0]], dtype=jnp.float32)
    x_std = jnp.sqrt(jnp.sum(jnp.square(x), axis=1, keepdims=True)) + 1e-8
    x = x / x_std
    sqrt_half = 1.0 / jnp.sqrt(2.0)
    normalized = jnp.array([[sqrt_half, sqrt_half], [-1, 0]])
    # If we normalized the wrong axis, it'd be [[sqrt_half 1] [-sqrt_half 0]]
    prop_better_than_random_permutation(
        x,
        jnp.array([[1, 0], [0, 0]], dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.uint32),
        jnp.array([False, False], dtype=jnp.bool),
    )


def test_better_than_random_permutation_frozen_2():
    prop_better_than_random_permutation(
        jnp.array([[0, 1], [0, 0]], dtype=jnp.float32),
        jnp.array([[0, 0], [0, -1]], dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.uint32),
        jnp.array([False, False], dtype=jnp.bool),
    )


def test_better_than_random_permutation_frozen_3():
    prop_better_than_random_permutation(
        jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=jnp.float32),
        jnp.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=jnp.float32),
        jnp.array([0, 1, 2], dtype=jnp.uint32),
        jnp.array([False, False, False], dtype=jnp.bool),
    )


@given(
    hnp.arrays(dtype=jnp.float32, shape=(2, 2)),
    hnp.arrays(dtype=jnp.float32, shape=(2, 2)),
    st.permutations(range(2)),
    hnp.arrays(dtype=jnp.bool, shape=(2,)),
)
def test_better_than_random_permutation_prop_2(x, y, randomly_chosen, flip):
    prop_better_than_random_permutation(
        jnp.array(x),
        jnp.array(y),
        jnp.array(randomly_chosen, dtype=jnp.uint32),
        jnp.array(flip, dtype=jnp.bool),
    )


@given(
    hnp.arrays(dtype=jnp.float32, shape=(3, 3)),
    hnp.arrays(dtype=jnp.float32, shape=(3, 3)),
    st.permutations(range(3)),
    hnp.arrays(dtype=jnp.bool, shape=(3,)),
)
def test_better_than_random_permutation_prop_3(x, y, randomly_chosen, flip):
    prop_better_than_random_permutation(
        jnp.array(x),
        jnp.array(y),
        jnp.array(randomly_chosen, dtype=jnp.uint32),
        jnp.array(flip, dtype=jnp.bool),
    )


@given(
    hnp.arrays(dtype=jnp.float32, shape=(4, 4)),
    hnp.arrays(dtype=jnp.float32, shape=(4, 4)),
    st.permutations(range(4)),
    hnp.arrays(dtype=jnp.bool, shape=(4,)),
)
def test_better_than_random_permutation_prop_4(x, y, randomly_chosen, flip):
    prop_better_than_random_permutation(
        jnp.array(x),
        jnp.array(y),
        jnp.array(randomly_chosen, dtype=jnp.uint32),
        jnp.array(flip, dtype=jnp.bool),
    )


@given(
    hnp.arrays(dtype=jnp.float32, shape=(5, 5)),
    hnp.arrays(dtype=jnp.float32, shape=(5, 5)),
    st.permutations(range(5)),
    hnp.arrays(dtype=jnp.bool, shape=(5,)),
)
def test_better_than_random_permutation_prop_5(x, y, randomly_chosen, flip):
    prop_better_than_random_permutation(
        jnp.array(x),
        jnp.array(y),
        jnp.array(randomly_chosen, dtype=jnp.uint32),
        jnp.array(flip, dtype=jnp.bool),
    )


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
    y = feedforward.run(w, x, nl=jnn.gelu)
    yp = feedforward.run(wp, x, nl=jnn.gelu)
    assert jnp.allclose(y, yp)


@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_1():
    prop_permute_hidden_layers(
        Weights(
            [jnp.zeros([5, 5]) for _ in range(3)], [jnp.zeros([5]) for _ in range(3)]
        ),
        [jnp.arange(5, dtype=jnp.uint32), jnp.arange(5, dtype=jnp.uint32)],
        jnp.zeros([1, 5]),
    )


@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_2():
    prop_permute_hidden_layers(
        Weights(
            list(jnp.full([3, 5, 5], 3.2994486e17)),
            list(jnp.full([3, 5], 5.442048e17)),
        ),
        p=[jnp.arange(5, dtype=jnp.uint32) for _ in range(2)],
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
    loss, ps = permutations.layer_distance(
        Weights(Wactual, Bactual),
        Weights(Wideal, Bideal),
    )
    assert len(ps) == 2
    assert jnp.isclose(loss, 0)
    assert jnp.all(ps[0].indices == perm)
    assert jnp.all(jnp.logical_not(ps[0].flip))
    assert jnp.all(ps[1].indices == inv_perm)
    assert jnp.all(jnp.logical_not(ps[1].flip))


def prop_optim_trivial(
    optim: Optimizer,
    opt_params: PyTree[Float[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float[Array, "..."]]],
        PyTree[Float[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(1.0, dtype=jnp.float32),
):
    x = jnp.array([[30], [-10]], dtype=jnp.float32)
    opt_state = opt_state_init(x, opt_params)
    loss = lambda x: jnp.sum(jnp.abs(10 - x))
    dLdx = jit(grad(loss))
    for _ in range(100):
        opt_state, x = jit(optim)(opt_params, opt_state, x, dLdx(x))
    assert loss(x) < 1


@jaxtyped(typechecker=beartype)
def test_optim_trivial():
    # Why all in one test?
    # Pytest seems to restart on every test, thus erasing JIT-compiled functions,
    # and this should significantly speed things up.

    print("SGD")
    import metaoptimizer.optimizers.sgd as optim

    prop_optim_trivial(optim.update, optim.defaults(lr=jnp.array(0.5)), optim.init)

    print("Weight decay")
    import metaoptimizer.optimizers.weight_decay as optim

    prop_optim_trivial(optim.update, optim.defaults(lr=jnp.array(0.5)), optim.init)

    print("Momentum")
    import metaoptimizer.optimizers.momentum as optim

    prop_optim_trivial(optim.update, optim.defaults(lr=jnp.array(0.5)), optim.init)

    print("Nesterov")
    import metaoptimizer.optimizers.nesterov as optim

    prop_optim_trivial(optim.update, optim.defaults(lr=jnp.array(0.5)), optim.init)

    print("RMSProp")
    import metaoptimizer.optimizers.rmsprop as optim

    prop_optim_trivial(optim.update, optim.defaults(lr=jnp.array(0.5)), optim.init)

    print("Adam")
    import metaoptimizer.optimizers.adam as optim

    prop_optim_trivial(optim.update, optim.defaults(lr=jnp.array(0.5)), optim.init)

    print("Swiss army knife")
    import metaoptimizer.optimizers.swiss_army_knife as optim

    prop_optim_trivial(optim.update, optim.defaults(lr=jnp.array(0.5)), optim.init)


NDIM = 3
BATCH = 32
LAYERS = 3
EPOCHS = 1
LR = jnp.array(0.001)


@jaxtyped(typechecker=beartype)
def prop_optim(
    optim: Optimizer,
    opt_params: PyTree[Float[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float[Array, "..."]]],
        PyTree[Float[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(1.0, dtype=jnp.float32),
):
    w = feedforward.init([NDIM for _ in range(LAYERS + 1)], jrnd.PRNGKey(42))
    key = jrnd.PRNGKey(42)
    orig_x = jrnd.normal(key, [BATCH, NDIM])
    forward_pass: ForwardPass = feedforward.run
    eval_weights = jit(
        lambda weights: checkify(training.loss, errors=all_checks)(
            weights, forward_pass, orig_x, jnp.sin(10 * orig_x), power
        )
    )
    err, orig_loss = eval_weights(w)
    err.throw()
    opt_state = opt_state_init(w, opt_params)
    for _ in range(EPOCHS):
        k, key = jrnd.split(key)
        x = jrnd.normal(k, [BATCH, NDIM])
        err, (w, opt_state, _) = training.step(
            w, forward_pass, x, jnp.sin(10 * x), optim, opt_params, opt_state, power
        )
        err.throw()
    err, post_loss = eval_weights(w)
    err.throw()
    # make sure we learned *something*:
    assert post_loss < orig_loss


@jaxtyped(typechecker=beartype)
def test_prop_optim():
    # Why all in one test?
    # Pytest seems to restart on every test, thus erasing JIT-compiled functions,
    # and this should significantly speed things up.

    print("SGD")
    import metaoptimizer.optimizers.sgd as optim

    prop_optim(optim.update, optim.defaults(lr=LR), optim.init)

    print("Weight decay")
    import metaoptimizer.optimizers.weight_decay as optim

    prop_optim(optim.update, optim.defaults(lr=LR), optim.init)

    print("Momentum")
    import metaoptimizer.optimizers.momentum as optim

    prop_optim(optim.update, optim.defaults(lr=LR), optim.init)

    print("Nesterov")
    import metaoptimizer.optimizers.nesterov as optim

    prop_optim(optim.update, optim.defaults(lr=LR), optim.init)

    print("RMSProp")
    import metaoptimizer.optimizers.rmsprop as optim

    prop_optim(optim.update, optim.defaults(lr=LR), optim.init)

    print("Adam")
    import metaoptimizer.optimizers.adam as optim

    prop_optim(optim.update, optim.defaults(lr=LR), optim.init)

    print("Swiss army knife")
    import metaoptimizer.optimizers.swiss_army_knife as optim

    prop_optim(optim.update, optim.defaults(lr=LR), optim.init)


@jaxtyped(typechecker=beartype)
def prop_optim_downhill(
    optim: Optimizer,
    opt_params: PyTree[Float[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float[Array, "..."]]],
        PyTree[Float[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(1.0, dtype=jnp.float32),
):
    # print(f"Initl optimizer parameters: {opt_params}")
    orig_opt_params = opt_params
    w = feedforward.init([NDIM for _ in range(LAYERS + 1)], jrnd.PRNGKey(42))
    key = jrnd.PRNGKey(42)
    orig_x = jrnd.normal(key, [BATCH, NDIM])
    forward_pass: ForwardPass = feedforward.run
    eval_weights = jit(
        lambda weights: checkify(training.loss_and_grad, errors=all_checks)(
            weights, forward_pass, orig_x, jnp.sin(10 * orig_x), power
        )
    )
    err, (orig_loss, last_dLdw) = eval_weights(w)
    err.throw()
    opt_state = opt_state_init(w, opt_params)
    for _ in range(EPOCHS):
        k, key = jrnd.split(key)
        x = jrnd.normal(k, [BATCH, NDIM])
        err, aux = training.step_downhill(
            w,
            forward_pass,
            x,
            jnp.sin(10 * x),
            optim,
            opt_params,
            opt_state,
            last_dLdw,
            power,
        )
        err.throw()
        w, opt_state, opt_params, _, last_dLdw = aux
        # print(f"Intrm optimizer parameters: {opt_params}")
    # print(f"Final optimizer parameters: {opt_params}")
    err, (post_loss, _) = eval_weights(w)
    err.throw()
    # make sure we learned *something*:
    assert post_loss < orig_loss
    assert not tree_reduce(
        operator.and_, tree_map(jnp.allclose, opt_params, orig_opt_params)
    )


@jaxtyped(typechecker=beartype)
def test_prop_optim_downhill():
    # Why all in one test?
    # Pytest seems to restart on every test, thus erasing JIT-compiled functions,
    # and this should significantly speed things up.

    print("SGD")
    import metaoptimizer.optimizers.sgd as optim

    prop_optim_downhill(optim.update, optim.defaults(lr=LR), optim.init)

    print("Weight decay")
    import metaoptimizer.optimizers.weight_decay as optim

    prop_optim_downhill(optim.update, optim.defaults(lr=LR), optim.init)

    print("Momentum")
    import metaoptimizer.optimizers.momentum as optim

    prop_optim_downhill(optim.update, optim.defaults(lr=LR), optim.init)

    print("Nesterov")
    import metaoptimizer.optimizers.nesterov as optim

    prop_optim_downhill(optim.update, optim.defaults(lr=LR), optim.init)

    print("RMSProp")
    import metaoptimizer.optimizers.rmsprop as optim

    prop_optim_downhill(optim.update, optim.defaults(lr=LR), optim.init)

    print("Adam")
    import metaoptimizer.optimizers.adam as optim

    prop_optim_downhill(optim.update, optim.defaults(lr=LR), optim.init)

    print("Swiss army knife")
    import metaoptimizer.optimizers.swiss_army_knife as optim

    prop_optim_downhill(optim.update, optim.defaults(lr=LR), optim.init)


@jaxtyped(typechecker=beartype)
def prop_optim_global(
    optim: Optimizer,
    opt_params: PyTree[Float[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float[Array, "..."]]],
        PyTree[Float[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(1.0, dtype=jnp.float32),
):
    # print(f"Initl optimizer parameters: {opt_params}")
    orig_opt_params = opt_params
    [w, w_ideal] = [
        feedforward.init(
            [NDIM for _ in range(LAYERS + 1)],
            jrnd.PRNGKey(42 + i),
        )
        for i in range(2)
    ]
    key = jrnd.PRNGKey(42)
    orig_x = jrnd.normal(key, [BATCH, NDIM])
    forward_pass: ForwardPass = feedforward.run
    jit_forward_pass = jit(checkify(forward_pass, errors=all_checks))
    err, orig_y = jit_forward_pass(w_ideal, orig_x)
    err.throw()
    eval_weights = jit(
        lambda weights: checkify(training.loss, errors=all_checks)(
            weights, forward_pass, orig_x, orig_y, power
        )
    )
    err, orig_loss = eval_weights(w)
    err.throw()
    orig_layer_distance, _ = permutations.layer_distance(w, w_ideal)
    opt_state = opt_state_init(w, opt_params)
    for _ in range(EPOCHS):
        k, key = jrnd.split(key)
        x = jrnd.normal(k, [BATCH, NDIM])
        err, y_ideal = jit_forward_pass(w_ideal, x)
        err.throw()
        err, (w, opt_state, opt_params, _, _) = training.step_global(
            w,
            forward_pass,
            x,
            y_ideal,
            optim,
            opt_params,
            opt_state,
            global_minimum=w_ideal,
            power=power,
        )
        err.throw()
        # print(f"Intrm optimizer parameters: {opt_params}")
    # print(f"Final optimizer parameters: {opt_params}")
    err, post_loss = eval_weights(w)
    err.throw()
    post_layer_distance, _ = permutations.layer_distance(w, w_ideal)
    # make sure we learned *something*:
    assert post_loss < orig_loss
    assert post_layer_distance < orig_layer_distance
    assert not tree_reduce(
        operator.and_, tree_map(jnp.allclose, opt_params, orig_opt_params)
    )


@jaxtyped(typechecker=beartype)
def test_prop_optim_global():
    # Why all in one test?
    # Pytest seems to restart on every test, thus erasing JIT-compiled functions,
    # and this should significantly speed things up.

    print("SGD")
    import metaoptimizer.optimizers.sgd as optim

    prop_optim_global(optim.update, optim.defaults(lr=LR), optim.init)

    print("Weight decay")
    import metaoptimizer.optimizers.weight_decay as optim

    prop_optim_global(optim.update, optim.defaults(lr=LR), optim.init)

    print("Momentum")
    import metaoptimizer.optimizers.momentum as optim

    prop_optim_global(optim.update, optim.defaults(lr=LR), optim.init)

    print("Nesterov")
    import metaoptimizer.optimizers.nesterov as optim

    prop_optim_global(optim.update, optim.defaults(lr=LR), optim.init)

    print("RMSProp")
    import metaoptimizer.optimizers.rmsprop as optim

    prop_optim_global(optim.update, optim.defaults(lr=LR), optim.init)

    print("Adam")
    import metaoptimizer.optimizers.adam as optim

    prop_optim_global(optim.update, optim.defaults(lr=LR), optim.init)

    print("Swiss army knife")
    import metaoptimizer.optimizers.swiss_army_knife as optim

    prop_optim_global(optim.update, optim.defaults(lr=LR), optim.init)
