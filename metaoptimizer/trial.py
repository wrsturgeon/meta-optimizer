from metaoptimizer import feedforward, permutations, training
from metaoptimizer.jit import jit
from metaoptimizer.permutations import Permutation
from metaoptimizer.training import ForwardPass, Optimizer
from metaoptimizer.weights import wb, Weights

from beartype import beartype
from beartype.typing import Callable, List, Optional, Tuple, TypeAlias
from importlib import import_module
from jax import debug, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental import io_callback
from jax.lax import fori_loop
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jaxtyping import jaxtyped, Array, Bool, Float32, Float64, PyTree, UInt16
import matplotlib.pyplot as plt
import operator
import os
import shutil
from time import time
from types import ModuleType


@jit(6, 7, 8, 9)
def step(
    key: Array,
    w: Weights,
    opt_state: PyTree[Float64[Array, "..."]],
    opt_params: PyTree[Float64[Array, ""]],
    w_ideal: Weights,
    power: Float32[Array, ""],
    batch: int,
    ndim: int,
    forward_pass: ForwardPass,
    optimizer: Optimizer,
) -> Tuple[
    Array,
    Weights,
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, ""]],
    List[Permutation],
    Float32[Array, ""],
]:
    k, key = jrnd.split(key)
    x = jrnd.normal(k, [batch, ndim], dtype=jnp.float32)
    y_ideal = forward_pass(w_ideal, x)
    w, opt_state, opt_params, permutation, L = training.step_global(
        w,
        forward_pass,
        x,
        y_ideal,
        optimizer,
        opt_params,
        opt_state,
        w_ideal,
        power,
    )
    return key, w, opt_state, opt_params, permutation, L


TrainOneStepOutput: TypeAlias = Tuple[
    Array,
    Weights,
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, ""]],
    Optional[Float32[Array, "n_iter"]],
    Optional[Float64[Array, "n_iter layers n_out n_in+1"]],
    Optional[UInt16[Array, "n_iter layers n_out"]],
    Optional[Bool[Array, "n_iter layers n_out"]],
    Optional[PyTree[Float64[Array, "n_iter"]]],
    Optional[Float64[Array, "n_iter layers n_out n_in"]],
    Optional[Float64[Array, "n_iter layers n_out"]],
]


@jit(6, 7, 8, 9)
def train_one_step(
    key: Array,
    w: Weights,
    opt_state: PyTree[Float64[Array, "..."]],
    opt_params: PyTree[Float64[Array, ""]],
    w_ideal: Weights,
    power: Float32[Array, ""],
    batch: int,
    ndim: int,
    forward_pass: ForwardPass,
    optimizer: Optimizer,
    losses: Optional[Float32[Array, "n_iter"]],
    weight_distances: Optional[Float64[Array, "n_iter layers n_out n_in+1"]],
    permutation_indices: Optional[UInt16[Array, "n_iter layers n_out"]],
    permutation_flips: Optional[Bool[Array, "n_iter layers n_out"]],
    opt_params_hist: Optional[PyTree[Float64[Array, "n_iter"]]],
    w_hist: Optional[Float64[Array, "n_iter layers n_out n_in"]],
    b_hist: Optional[Float64[Array, "n_iter layers n_out"]],
    i: UInt16[Array, ""],
) -> TrainOneStepOutput:

    key, w, opt_state, opt_params, permutation, L = step(
        key,
        w,
        opt_state,
        opt_params,
        w_ideal,
        power,
        batch,
        ndim,
        forward_pass,
        optimizer,
    )

    if losses is not None:
        losses = losses.at[i].set(jnp.array(L))

    if weight_distances is not None:
        wb_actual = wb(w)
        wb_ideal = wb(w_ideal)
        weight_distances = weight_distances.at[i].set(
            jnp.sum(jnp.abs(wb_actual - wb_ideal))
        )

    if permutation_indices is not None:
        permutation_indices = permutation_indices.at[i].set(
            jnp.array(permutation.indices)
        )

    if permutation_flips is not None:
        permutation_flips = permutation_flips.at[i].set(jnp.array(permutation.flip))

    if opt_params_hist is not None:
        opt_params_hist = tree_map(
            lambda x, y: x.at[i].set(y),
            opt_params_hist,
            opt_params,
        )

    if w_hist is not None:
        w_hist = w_hist.at[i].set(w.W)

    if b_hist is not None:
        b_hist = b_hist.at[i].set(w.B)

    return (
        key,
        w,
        opt_state,
        opt_params,
        losses,
        weight_distances,
        permutation_indices,
        permutation_flips,
        opt_params_hist,
        w_hist,
        b_hist,
    )


@jit
def train_contiguous(
    key: Array,
    w: Weights,
    opt_state: PyTree[Float64[Array, "..."]],
    opt_params: PyTree[Float64[Array, ""]],
    w_ideal: Weights,
    power: Float32[Array, ""],
    batch: int,
    ndim: int,
    forward_pass: ForwardPass,
    optimizer: Optimizer,
    how_many_iterations: UInt16[Array, ""],
) -> TrainOneStepOutput:

    def fold(i: UInt16[Array, ""], package: TrainOneStepOutput) -> TrainOneStepOutput:

        (
            key,
            w,
            opt_state,
            opt_params,
            losses,
            weight_distances,
            permutation_indices,
            permutation_flips,
            opt_params_hist,
            w_hist,
            b_hist,
        ) = package

        return train_one_step(
            key,
            w,
            opt_state,
            opt_params,
            w_ideal,
            power,
            batch,
            ndim,
            forward_pass,
            optimizer,
            losses,
            weight_distances,
            permutation_indices,
            permutation_flips,
            opt_params_hist,
            w_hist,
            b_hist,
            i,
        )

    # NOTE: `fori_loop` does not unroll! (this is good)
    return fori_loop(
        0,
        how_many_iterations,
        fold,
        (
            key,
            w,
            opt_state,
            opt_params,
            jnp.empty([how_many_iterations], dtype=jnp.float32),
            jnp.empty([how_many_iterations], dtype=jnp.float32),
            jnp.empty([how_many_iterations], dtype=jnp.uint16),
            jnp.empty([how_many_iterations], dtype=jnp.bool),
            jnp.empty([how_many_iterations], dtype=jnp.float32),
            jnp.empty([how_many_iterations], dtype=jnp.float32),
            jnp.empty([how_many_iterations], dtype=jnp.float32),
        ),
    )


@jit
def train_percentages(
    key: Array,
    w: Weights,
    opt_state: PyTree[Float64[Array, "..."]],
    opt_params: PyTree[Float64[Array, ""]],
    w_ideal: Weights,
    power: Float32[Array, ""],
    batch: int,
    ndim: int,
    forward_pass: ForwardPass,
    optimizer: Optimizer,
    how_many_iterations: UInt16[Array, ""],
) -> Tuple[
    Optional[Float32[Array, "n_iter"]],
    Optional[Float64[Array, "n_iter layers n_out n_in+1"]],
    Optional[UInt16[Array, "n_iter layers n_out"]],
    Optional[Bool[Array, "n_iter layers n_out"]],
    Optional[PyTree[Float64[Array, "n_iter"]]],
    Optional[Float64[Array, "n_iter layers n_out n_in"]],
    Optional[Float64[Array, "n_iter layers n_out"]],
]:

    pct_size = how_many_iterations // 100
    assert pct_size * 100 == how_many_iterations  # i.e., cleanly divisible

    def fold(i: UInt16[Array, ""], package: TrainOneStepOutput) -> TrainOneStepOutput:

        (
            key,
            w,
            opt_state,
            opt_params,
            losses,
            weight_distances,
            permutation_indices,
            permutation_flips,
            opt_params_hist,
            w_hist,
            b_hist,
        ) = package

        (
            key,
            w,
            opt_state,
            opt_params,
            pct_losses,
            pct_weight_distances,
            pct_permutation_indices,
            pct_permutation_flips,
            pct_opt_params_hist,
            pct_w_hist,
            pct_b_hist,
        ) = train_contiguous(
            key,
            w,
            opt_state,
            opt_params,
            w_ideal,
            power,
            batch,
            ndim,
            forward_pass,
            optimizer,
            pct_size,
        )

        if losses is not None:
            losses.at[i].set(pct_losses)
        if weight_distances is not None:
            weight_distances.at[i].set(pct_weight_distances)
        if permutation_indices is not None:
            permutation_indices.at[i].set(pct_permutation_indices)
        if permutation_flips is not None:
            permutation_flips.at[i].set(pct_permutation_flips)
        if opt_params_hist is not None:
            opt_params_hist.at[i].set(pct_opt_params_hist)
        if w_hist is not None:
            w_hist.at[i].set(pct_w_hist)
        if b_hist is not None:
            b_hist.at[i].set(pct_b_hist)

        debug.print("{j}%", j=(i + 1))

        return (
            key,
            w,
            opt_state,
            opt_params,
            losses,
            weight_distances,
            permutation_indices,
            permutation_flips,
            opt_params_hist,
            w_hist,
            b_hist,
        )

    # NOTE: `fori_loop` does not unroll! (this is good)
    (
        key,
        w,
        opt_state,
        opt_params,
        losses,
        weight_distances,
        permutation_indices,
        permutation_flips,
        opt_params_hist,
        w_hist,
        b_hist,
    ) = fori_loop(
        0,
        100,
        fold,
        (
            key,
            w,
            opt_state,
            opt_params,
            losses,
            weight_distances,
            permutation_indices,
            permutation_flips,
            opt_params_hist,
            w_hist,
            b_hist,
        ),
    )

    return (
        losses,
        weight_distances,
        permutation_indices,
        permutation_flips,
        opt_params_hist,
        w_hist,
        b_hist,
    )


@jit
def run(
    key: Array,
    ndim: int = 3,
    batch: int = 1,
    layers: int = 2,
    nonlinearity: Callable[[Float32[Array, "*n"]], Float32[Array, "*n"]] = jnn.gelu,
    optim: ModuleType = import_module("metaoptimizer.optimizers.sgd"),
    training_steps: int = 100000,
    subdir: List[str] = ["logs"],
    convergence_only: bool = False,
    power: Float32[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
    lr: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
    initial_distance: Float64[Array, ""] = jnp.array(0.1, dtype=jnp.float64),
    prefix: str = "",
) -> None:

    debug.print(prefix + "Setting up the model architecture...")

    k1, k2 = jrnd.split(key)

    # Weight initialization (note `w_ideal` is really the *goal*)
    shapes = tuple([ndim for _ in range(layers + 1)])
    w_ideal = feedforward.init(shapes, k1, True)

    # Uncomment if you want `w` to start already very close to `w_ideal`:
    w_flat, w_def = tree_flatten(w_ideal)
    w_keys = tree_unflatten(w_def, jrnd.split(k2, len(w_flat)))
    w = tree_map(
        lambda x, k: x + initial_distance * jrnd.normal(k, x.shape), w_ideal, w_keys
    )

    # Optimizer initialization
    # TODO: Is there a Pythonic way to reduce redundancy here?
    optimizer = optim.update
    opt_params = optim.defaults(lr=lr)
    opt_state = optim.init(w, opt_params)

    # Forward pass initialization
    forward_pass: ForwardPass = lambda weights, x: feedforward.run(
        weights,
        x,
        nonlinearity,
    )

    # Replicable pseudorandomness
    key = jrnd.PRNGKey(42)  # the answer

    # Record-keeping
    weight_distances = [
        [jnp.empty([], dtype=jnp.float32) for _ in range(training_steps)]
        for _ in range(layers)
    ]
    nonterminal_layers = layers - 1
    if not convergence_only:
        losses = jnp.empty([training_steps], dtype=jnp.float32)
        permutation_indices = [
            jnp.empty([training_steps, ndim], dtype=jnp.uint16)
            for _ in range(nonterminal_layers)
        ]
        permutation_flips = [
            jnp.empty([training_steps, ndim], dtype=bool)
            for _ in range(nonterminal_layers)
        ]
        opt_params_hist: List[PyTree[Float64[Array, ""]]] = []
        w_hist: List[PyTree[Float64[Array, "..."]]] = []
        w_ideal_hist: List[PyTree[Float64[Array, "..."]]] = []

    debug.print(prefix + "Compiling the training loop...")

    # Training loop:
    t0 = time()
    (
        losses,
        weight_distances,
        permutation_indices,
        permutation_flips,
        opt_params_hist,
        w_hist,
        b_hist,
    ) = train_percentages(
        key,
        w,
        opt_state,
        opt_params,
        w_ideal,
        power,
        batch,
        ndim,
        forward_pass,
        optimizer,
        training_steps,
    )
    debug.print(prefix + f"Done in {int(time() - t0)} seconds")

    debug.print(prefix + "Saving...")

    cwd = os.getcwd()
    cwd = os.path.join(cwd, *subdir)
    path = lambda *args: os.path.join(cwd, *args)

    @jaxtyped(typechecker=beartype)
    def save(array: Array, *args) -> None:
        io_callback(
            lambda x: jnp.save(
                path(*args),
                x,
                allow_pickle=False,
            ),
            None,
            array,
        )

    if os.path.exists(path()):
        shutil.rmtree(path(), ignore_errors=True)
    os.makedirs(path("weight_distances"))
    if not convergence_only:
        save(losses, "losses.npy")
        os.makedirs(path("optimizer"))
        os.makedirs(path("permutations"))
        os.makedirs(path("weights"))

    w_ideal_permuted = permutations.permute_hidden_layers(w_ideal, permutation)

    for i in range(layers):

        save(
            jnp.array(weight_distances[i]),
            "weight_distances",
            f"layer_{i}.npy",
        )

        if convergence_only:
            save(
                jnp.less(
                    jnp.sum(jnp.array(weight_distances[-1])),
                    jnp.sum(jnp.array(weight_distances[0])),
                ),
                "closer_or_not.npy",
            )

        else:
            os.makedirs(path("weights", f"layer_{i}", "weights"))
            os.makedirs(path("weights", f"layer_{i}", "biases"))
            save(
                jnp.array([jnp.ravel(wh.W[i]) for wh in w_hist]),
                "weights",
                f"layer_{i}",
                "weights",
                "w_historical.npy",
            )
            save(
                jnp.array([jnp.ravel(wh.B[i]) for wh in w_hist]),
                "weights",
                f"layer_{i}",
                "biases",
                "w_historical.npy",
            )
            save(
                jnp.array([jnp.ravel(wh.W[i]) for wh in w_ideal_hist]),
                "weights",
                f"layer_{i}",
                "weights",
                "w_ideal_historical.npy",
            )
            save(
                jnp.array([jnp.ravel(wh.B[i]) for wh in w_ideal_hist]),
                "weights",
                f"layer_{i}",
                "biases",
                "w_ideal_historical.npy",
            )
            save(
                w_ideal.W[i],
                "weights",
                f"layer_{i}",
                "weights",
                "ideal_orig.npy",
            )
            save(
                w_ideal_permuted.W[i],
                "weights",
                f"layer_{i}",
                "weights",
                "ideal_perm.npy",
            )
            save(
                w.W[i],
                "weights",
                f"layer_{i}",
                "weights",
                "final.npy",
            )
            save(
                w_ideal.B[i],
                "weights",
                f"layer_{i}",
                "biases",
                "ideal_orig.npy",
            )
            save(
                w_ideal_permuted.B[i],
                "weights",
                f"layer_{i}",
                "biases",
                "ideal_perm.npy",
            )
            save(
                w.B[i],
                "weights",
                f"layer_{i}",
                "biases",
                "final.npy",
            )

    if not convergence_only:
        for i in range(nonterminal_layers):
            save(
                permutation_indices[i],
                "permutations",
                f"layer_{i}_indices.npy",
            )
            save(
                permutation_flips[i],
                "permutations",
                f"layer_{i}_flips.npy",
            )
        if hasattr(opt_params_hist[0], "log_lr"):
            save(
                jnp.array([jnp.exp(p.log_lr) for p in opt_params_hist]),
                "optimizer",
                "lr.npy",
            )
        if hasattr(opt_params_hist[0], "inv_sig_moving_average_decay"):
            save(
                jnp.array(
                    [
                        jnn.sigmoid(p.inv_sig_moving_average_decay)
                        for p in opt_params_hist
                    ]
                ),
                "optimizer",
                "moving_average_decay.npy",
            )
        if hasattr(opt_params_hist[0], "inv_sig_moving_square_decay"):
            save(
                jnp.array(
                    [
                        jnn.sigmoid(p.inv_sig_moving_square_decay)
                        for p in opt_params_hist
                    ]
                ),
                "optimizer",
                "moving_square_decay.npy",
            )
        if hasattr(opt_params_hist[0], "inv_sig_moving_square_quotient"):
            save(
                jnp.array(
                    [
                        jnn.sigmoid(p.inv_sig_moving_square_quotient)
                        for p in opt_params_hist
                    ]
                ),
                "optimizer",
                "moving_square_quotient.npy",
            )
        if hasattr(opt_params_hist[0], "inv_sig_momentum"):
            save(
                jnp.array([jnn.sigmoid(p.inv_sig_momentum) for p in opt_params_hist]),
                "optimizer",
                "momentum.npy",
            )
        if hasattr(opt_params_hist[0], "log_overstep"):
            save(
                jnp.array([jnp.exp(p.log_overstep) for p in opt_params_hist]),
                "optimizer",
                "overstep.npy",
            )
        if hasattr(opt_params_hist[0], "inv_sig_weight_decay"):
            save(
                jnp.array(
                    [jnn.sigmoid(p.inv_sig_weight_decay) for p in opt_params_hist]
                ),
                "optimizer",
                "weight_decay.npy",
            )
        if hasattr(opt_params_hist[0], "log_epsilon"):
            save(
                jnp.array([jnp.exp(p.log_epsilon) for p in opt_params_hist]),
                "optimizer",
                "epsilon.npy",
            )
