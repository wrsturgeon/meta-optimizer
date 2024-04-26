from os import environ


from metaoptimizer import (
    feedforward,
    permutations,
    training,
)
from metaoptimizer.optimizers import Optimizer
from metaoptimizer.permutations import Permutation
from metaoptimizer.training import ForwardPass
from metaoptimizer.weights import layers, wb, Weights

from beartype import beartype
from beartype.typing import Any, Callable, Iterable, Protocol, Tuple
from hypothesis import given, reproduce_failure, settings, strategies as st, Verbosity
from hypothesis.extra import numpy as hnp
from jax import jit, grad, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental.checkify import all_checks, checkify
from jax.lax import stop_gradient
from jax.numpy import linalg as jla
from jax.tree_util import tree_map, tree_reduce, tree_structure
from jaxtyping import (
    jaxtyped,
    Array,
    Bool,
    Float,
    Float64,
    PyTree,
    TypeCheckError,
    UInt32,
)
from math import prod
from numpy.typing import ArrayLike
import operator
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
    print("*** Running in CI mode")
    settings.load_profile("ci")
    TEST_COUNT = TEST_COUNT_CI
else:  # pragma: no cover
    print(
        f'*** NOT running in CI mode (environment variable `GITHUB_CI` is `{gh_ci}`, which is not `"1"`)'
    )
    settings.load_profile("no_deadline")
    TEST_COUNT = TEST_COUNT_NORMAL


@jaxtyped(typechecker=beartype)
def no_flip(indices_list: Iterable[int]) -> Permutation:
    indices = jnp.array(indices_list, dtype=jnp.uint32)
    return Permutation(
        indices=indices,
        flip=jnp.full_like(indices, False, dtype=jnp.bool),
    )


NDIM = 3
BATCH = 32
LAYERS = 3
EPOCHS = 1
LR = jnp.array(0.001)


@given(hnp.arrays(dtype=jnp.float32, shape=(3, 3)))
@jaxtyped(typechecker=beartype)
def test_feedforward_id_prop(np_x: ArrayLike) -> None:
    x = jnp.array(np_x)
    if not jnp.all(jnp.isfinite(x)):
        return
    y = feedforward.run(
        Weights(jnp.eye(3, 3)[jnp.newaxis], jnp.zeros([3])[jnp.newaxis]),
        x,
        lambda z: z,
    )
    assert jnp.allclose(y, x)


# NOTE: The big problem with using rotation matrices is that,
# with practically all nonlinearities (e.g. ReLU or GELU),
# negative values are effectively eliminated whereas
# positive values are allowed to pass effectively unchanged.
# Rotation matrices use negative values extensively, but
# negation significantly (very significantly!) alters behavior.


# TODO: test varying dimensionality across layers


@jaxtyped(typechecker=beartype)
def test_permute_1() -> None:
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
    i = no_flip([3, 1, 4, 2, 0])
    y = permutations.permute(x, i, 0)
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
def test_permute_2() -> None:
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
    i = no_flip([3, 1, 4, 2, 0])
    y = permutations.permute(x, i, 1)
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


def test_permute_3() -> None:
    x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.float32)
    i = no_flip([2, 1, 0])
    assert jnp.allclose(
        permutations.permute(x, i, 0),
        jnp.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=jnp.float32),
    )
    assert jnp.allclose(
        permutations.permute(x, i, 1),
        jnp.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]], dtype=jnp.float32),
    )


@jaxtyped(typechecker=beartype)
def test_permute_size_check() -> None:
    x = jnp.eye(5, 5, dtype=jnp.float32)
    i = no_flip(range(6))
    with pytest.raises(AssertionError):
        permutations.permute(x, i, 0)


@jaxtyped(typechecker=beartype)
def test_permute_axis_check() -> None:
    x = jnp.eye(5, 5, dtype=jnp.float32)
    i = no_flip(range(5))
    with pytest.raises(IndexError):
        permutations.permute(x, i, 5)


@jaxtyped(typechecker=beartype)
def test_cut_axes() -> None:
    a = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
    )
    assert jnp.allclose(
        permutations.cut_axes(a, jnp.arange(4, dtype=jnp.uint32), 0),
        jnp.array(
            [
                [
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                [
                    [1, 2, 3, 4],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [13, 14, 15, 16],
                ],
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                ],
            ],
        ),
    )
    assert jnp.allclose(
        permutations.cut_axes(a, jnp.arange(4, dtype=jnp.uint32), 1),
        jnp.array(
            [
                [
                    [2, 3, 4],
                    [6, 7, 8],
                    [10, 11, 12],
                    [14, 15, 16],
                ],
                [
                    [1, 3, 4],
                    [5, 7, 8],
                    [9, 11, 12],
                    [13, 15, 16],
                ],
                [
                    [1, 2, 4],
                    [5, 6, 8],
                    [9, 10, 12],
                    [13, 14, 16],
                ],
                [
                    [1, 2, 3],
                    [5, 6, 7],
                    [9, 10, 11],
                    [13, 14, 15],
                ],
            ]
        ),
    )


@jaxtyped(typechecker=beartype)
def test_find_permutation_1() -> None:
    ideal = Weights(
        W=jnp.eye(5, 5, dtype=jnp.float64)[jnp.newaxis],
        B=jnp.ones([1, 5], dtype=jnp.float64),
    )
    actual = Weights(
        W=jnp.array(
            [
                [
                    [0, 0, -1, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ]
            ],
            dtype=jnp.float64,
        ),
        B=jnp.array([[-1, 1, -1, 1, 1]], dtype=jnp.float64),
    )
    wb_a = wb(actual)
    wb_i = wb(ideal)
    assert wb_a.shape[0] == 1
    assert wb_i.shape[0] == 1
    p = permutations.find_permutation(wb_a[0], wb_i[0])
    ideal_indices = jnp.array([2, 0, 3, 1, 4], dtype=jnp.uint32)
    ideal_flip = jnp.array([True, False, True, False, False], dtype=jnp.bool)
    assert jnp.all(p.indices == ideal_indices), f"{p.indices} =/= {ideal_indices}"
    assert jnp.all(p.flip == ideal_flip), f"{p.flip} =/= {ideal_flip}"


@jaxtyped(typechecker=beartype)
def prop_better_than_random_permutation(
    x: Float[Array, "n m"],
    y: Float[Array, "n m"],
    randomly_chosen_indices: UInt32[Array, "n"],
    randomly_chosen_flip: Bool[Array, "n"],
) -> None:
    if not (jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y))):
        return
    allegedly_ideal = permutations.find_permutation(x, y)
    randomly_chosen = Permutation(
        indices=randomly_chosen_indices,
        flip=randomly_chosen_flip,
    )
    x = x / (
        jnp.sqrt(jnp.sum(jnp.square(stop_gradient(x)), axis=1, keepdims=True)) + 1e-8
    )
    y = y / (
        jnp.sqrt(jnp.sum(jnp.square(stop_gradient(y)), axis=1, keepdims=True)) + 1e-8
    )
    ap = permutations.permute(y, allegedly_ideal, 0)
    rp = permutations.permute(y, randomly_chosen, 0)
    aL = jnp.sum(jnp.abs(ap - x))
    rL = jnp.sum(jnp.abs(rp - x))
    assert (jnp.abs(aL - rL) < 0.0001) or (aL < rL), f"{aL} </= {rL}"


def test_better_than_random_permutation_frozen_1() -> None:
    # First, test that we're normalizing the right axes:
    x = jnp.array([[1, 1], [-1, 0]], dtype=jnp.float32)
    x_std = (
        jnp.sqrt(jnp.sum(jnp.square(stop_gradient(x)), axis=1, keepdims=True)) + 1e-8
    )
    x = x / (x_std + 1e-8)
    sqrt_half = jnp.sqrt(0.5)
    normalized = jnp.array([[sqrt_half, sqrt_half], [-1, 0]])
    # If we normalized the wrong axis, it'd be [[sqrt_half 1] [-sqrt_half 0]]
    prop_better_than_random_permutation(
        x,
        jnp.array([[1, 0], [0, 0]], dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.uint32),
        jnp.array([False, False], dtype=jnp.bool),
    )


def test_better_than_random_permutation_frozen_2() -> None:
    prop_better_than_random_permutation(
        jnp.array([[0, 1], [0, 0]], dtype=jnp.float32),
        jnp.array([[0, 0], [0, -1]], dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.uint32),
        jnp.array([False, False], dtype=jnp.bool),
    )


def test_better_than_random_permutation_frozen_3() -> None:
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
def test_better_than_random_permutation_prop_2(x, y, randomly_chosen, flip) -> None:
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
def test_better_than_random_permutation_prop_3(x, y, randomly_chosen, flip) -> None:
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
def test_better_than_random_permutation_prop_4(x, y, randomly_chosen, flip) -> None:
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
def test_better_than_random_permutation_prop_5(x, y, randomly_chosen, flip) -> None:
    prop_better_than_random_permutation(
        jnp.array(x),
        jnp.array(y),
        jnp.array(randomly_chosen, dtype=jnp.uint32),
        jnp.array(flip, dtype=jnp.bool),
    )


@jaxtyped(typechecker=beartype)
def prop_permute_hidden_layers(
    w: Weights,
    p: list[Permutation],
    x: Float[Array, "batch ndim"],
    nl: Callable = jnn.gelu,
) -> None:
    assert layers(w) == len(p) + 1, f"{layers(w)} =/= {len(p)} + 1"
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
    print("wp")
    print(wp)
    print()
    y = feedforward.run(w, x, nl)
    yp = feedforward.run(wp, x, nl)
    assert jnp.allclose(y, yp), f"\n{y}\n=/=\n{yp}"


@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_1() -> None:
    prop_permute_hidden_layers(
        Weights(jnp.zeros([3, 5, 5]), jnp.zeros([3, 5])),
        [no_flip(range(5)) for _ in range(2)],
        jnp.zeros([1, 5], dtype=jnp.float32),
    )


@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_2() -> None:
    prop_permute_hidden_layers(
        w=Weights(
            W=jnp.ones([3, 5, 5], dtype=jnp.float64),
            B=jnp.ones([3, 5], dtype=jnp.float64),
        ),
        p=[
            no_flip(range(5)),
            Permutation(
                indices=jnp.arange(5, dtype=jnp.uint32),
                flip=jnp.array([False, False, False, False, True], dtype=jnp.bool),
            ),
        ],
        x=jnp.zeros([1, 5], dtype=jnp.float32),
    )


@given(
    hnp.arrays(dtype=jnp.float64, shape=(2, 5, 5)),
    hnp.arrays(dtype=jnp.float64, shape=(2, 5)),
    st.lists(st.permutations(range(5)), min_size=1, max_size=1),
    st.lists(st.lists(st.booleans(), min_size=5, max_size=5), min_size=1, max_size=1),
    hnp.arrays(dtype=jnp.float32, shape=(1, 5)),
)
@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_prop_1(
    W: ArrayLike,
    B: ArrayLike,
    P: list[list[int]],
    F: list[list[bool]],
    x: ArrayLike,
) -> None:
    prop_permute_hidden_layers(
        Weights(jnp.array(W, dtype=jnp.float64), jnp.array(B, dtype=jnp.float64)),
        [
            Permutation(
                indices=jnp.array(p, dtype=jnp.uint32),
                flip=jnp.array(f, dtype=jnp.bool),
            )
            for p, f in zip(P, F)
        ],
        jnp.array(x, dtype=jnp.float32),
    )


@reproduce_failure("6.99.12", b"AXicY2CAA0YG7GwsIgAAjQAE")
@given(
    hnp.arrays(dtype=jnp.float64, shape=(3, 5, 5)),
    hnp.arrays(dtype=jnp.float64, shape=(3, 5)),
    st.lists(st.permutations(range(5)), min_size=2, max_size=2),
    st.lists(st.lists(st.booleans(), min_size=5, max_size=5), min_size=2, max_size=2),
    hnp.arrays(dtype=jnp.float32, shape=(1, 5)),
)
@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_prop_2(
    W: ArrayLike,
    B: ArrayLike,
    P: list[list[int]],
    F: list[list[bool]],
    x: ArrayLike,
) -> None:
    prop_permute_hidden_layers(
        Weights(jnp.array(W, dtype=jnp.float64), jnp.array(B, dtype=jnp.float64)),
        [
            Permutation(
                indices=jnp.array(p, dtype=jnp.uint32),
                flip=jnp.array(f, dtype=jnp.bool),
            )
            for p, f in zip(P, F)
        ],
        jnp.array(x, dtype=jnp.float32),
    )


@given(
    hnp.arrays(dtype=jnp.float64, shape=(4, 5, 5)),
    hnp.arrays(dtype=jnp.float64, shape=(4, 5)),
    st.lists(st.permutations(range(5)), min_size=3, max_size=3),
    st.lists(st.lists(st.booleans(), min_size=5, max_size=5), min_size=3, max_size=3),
    hnp.arrays(dtype=jnp.float32, shape=(1, 5)),
)
@jaxtyped(typechecker=beartype)
def test_permute_hidden_layers_prop_3(
    W: ArrayLike,
    B: ArrayLike,
    P: list[list[int]],
    F: list[list[bool]],
    x: ArrayLike,
) -> None:
    prop_permute_hidden_layers(
        Weights(jnp.array(W, dtype=jnp.float64), jnp.array(B, dtype=jnp.float64)),
        [
            Permutation(
                indices=jnp.array(p, dtype=jnp.uint32),
                flip=jnp.array(f, dtype=jnp.bool),
            )
            for p, f in zip(P, F)
        ],
        jnp.array(x, dtype=jnp.float32),
    )


@jaxtyped(typechecker=beartype)
def test_layer_distance() -> None:
    eye = jnp.eye(5, 5)
    perm = no_flip([3, 1, 4, 2, 0])
    inv_perm = no_flip([4, 1, 3, 0, 2])
    Wactual = jnp.stack(
        [
            permutations.permute(eye, perm, 0),
            permutations.permute(eye, inv_perm, 0),
            eye,
        ]
    )
    Bactual = jnp.zeros([3, 5])
    Wideal = jnp.stack([eye, eye, eye])
    Bideal = jnp.zeros([3, 5])
    loss, ps = permutations.layer_distance(
        Weights(Wactual, Bactual),
        Weights(Wideal, Bideal),
    )
    assert len(ps) == 2, f"{len(ps)} =/= 2"
    assert jnp.isclose(loss, 0), f"{loss} =/= 0"


def prop_optim_trivial(
    optim: Optimizer,
    opt_params: PyTree[Float64[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float64[Array, "..."]]],
        PyTree[Float64[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
) -> None:
    x = jnp.array([[30], [-10]], dtype=jnp.float32)
    opt_state = opt_state_init(x, opt_params)
    loss = lambda x: jnp.sum(jnp.abs(10 - x))
    dLdx = jit(grad(loss))
    for _ in range(100):
        opt_state, x = jit(optim)(opt_params, opt_state, x, dLdx(x))
    assert loss(x) < 1


@jaxtyped(typechecker=beartype)
def test_optim_trivial() -> None:
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


@jaxtyped(typechecker=beartype)
def prop_optim(
    optim: Optimizer,
    opt_params: PyTree[Float64[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float64[Array, "..."]]],
        PyTree[Float64[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
) -> None:
    shapes = tuple([NDIM for _ in range(LAYERS + 1)])
    w = feedforward.init(shapes, jrnd.PRNGKey(42), False)
    key = jrnd.PRNGKey(42)
    orig_x = jrnd.normal(key, [BATCH, NDIM], dtype=jnp.float32)
    forward_pass = feedforward.run
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
        x = jrnd.normal(k, [BATCH, NDIM], dtype=jnp.float32)
        w, opt_state, _ = training.step(
            w, forward_pass, x, jnp.sin(10 * x), optim, opt_params, opt_state, power
        )
        err.throw()
    err, post_loss = eval_weights(w)
    err.throw()
    # make sure we learned *something*:
    assert post_loss < orig_loss


@jaxtyped(typechecker=beartype)
def test_prop_optim() -> None:
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
    opt_params: PyTree[Float64[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float64[Array, "..."]]],
        PyTree[Float64[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
) -> None:
    # print(f"Initl optimizer parameters: {opt_params}")
    orig_opt_params = opt_params
    shapes = tuple([NDIM for _ in range(LAYERS + 1)])
    w = feedforward.init(shapes, jrnd.PRNGKey(42), True)
    key = jrnd.PRNGKey(42)
    orig_x = jrnd.normal(key, [BATCH, NDIM], dtype=jnp.float32)
    forward_pass = feedforward.run
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
        x = jrnd.normal(k, [BATCH, NDIM], dtype=jnp.float32)
        aux = training.step_downhill(
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
def test_prop_optim_downhill() -> None:
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
    opt_params: PyTree[Float64[Array, ""]],
    opt_state_init: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Float64[Array, "..."]]],
        PyTree[Float64[Array, "..."]],
    ],
    power: Float[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
) -> None:
    # print(f"Initl optimizer parameters: {opt_params}")
    orig_opt_params = opt_params
    [w, w_ideal] = [
        feedforward.init(
            tuple([NDIM for _ in range(LAYERS + 1)]),
            jrnd.PRNGKey(42 + i),
            True,
        )
        for i in range(2)
    ]
    key = jrnd.PRNGKey(42)
    orig_x = jrnd.normal(key, [BATCH, NDIM], dtype=jnp.float32)
    forward_pass = feedforward.run
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
        x = jrnd.normal(k, [BATCH, NDIM], dtype=jnp.float32)
        err, y_ideal = jit_forward_pass(w_ideal, x)
        err.throw()
        w, opt_state, opt_params, _, _ = training.step_global(
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
def test_prop_optim_global() -> None:
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
