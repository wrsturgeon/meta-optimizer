from metaoptimizer import feedforward, permutations, training
from metaoptimizer.permutations import Permutation
from metaoptimizer.training import ForwardPass, Optimizer
from metaoptimizer.weights import layers, wb, Weights

from beartype import beartype
from beartype.typing import Callable, List, Optional, Tuple, TypeAlias
from check_and_compile import check_and_compile
from importlib import import_module
from jax import debug, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental import io_callback
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jaxtyping import (
    jaxtyped,
    Array,
    Bool,
    Float32,
    Float64,
    Int64,
    PyTree,
    UInt32,
    UInt64,
)
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import operator
import os
import shutil
from time import time
from types import ModuleType


@check_and_compile(6, 7, 8, 9)
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


@jaxtyped(typechecker=beartype)
def run(
    key: Array,
    ndim: int,
    batch: int,
    layers: int,
    forward_pass: Callable,
    optimizer: Optimizer,
    opt_state: PyTree[Float64[Array, "..."]],
    opt_params: PyTree[Float64[Array, ""]],
    nonlinearity: Callable[[Float32[Array, "*n"]], Float32[Array, "*n"]] = jnn.gelu,
    training_steps: int = 100000,
    subdir: Tuple = ("logs",),
    power: Float32[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
    initial_distance: Float64[Array, ""] = jnp.array(0.1, dtype=jnp.float64),
    prefix: str = "",
    track_convergence: bool = True,
    track_losses: bool = True,
    track_weight_distances: bool = True,
    track_permutation_indices: bool = True,
    track_permutation_flips: bool = True,
    track_opt_params_hist: bool = True,
    track_w_hist: bool = True,
    track_b_hist: bool = True,
    track_w_ideal_hist: bool = True,
    track_b_ideal_hist: bool = True,
    verbose: bool = True,
) -> None:

    if verbose:
        print(prefix + "Setting up the model architecture...")

    k1, k2 = jrnd.split(key)

    # Weight initialization (note `w_ideal` is really the *goal*)
    shapes = tuple([ndim for _ in range(layers + 1)])
    w_ideal = feedforward.init(shapes, k1, True)

    # Uncomment if you want `w` to start already very close to `w_ideal`:
    w_flat, w_def = tree_flatten(w_ideal)
    w_keys = tree_unflatten(w_def, jrnd.split(k2, len(w_flat)))
    w = tree_map(
        lambda x, k: x + initial_distance * jrnd.normal(k, x.shape),
        w_ideal,
        w_keys,
    )

    # Replicable pseudorandomness
    key = jrnd.PRNGKey(42)  # the answer

    # Record-keeping
    losses = (
        np.empty(
            [training_steps],
            dtype=np.float32,
        )
        if track_losses
        else None
    )
    weight_distances = (
        np.empty(
            [training_steps, layers],
            dtype=np.float32,
        )
        if track_weight_distances
        else None
    )
    permutation_indices = (
        np.empty(
            [training_steps, layers - 1, ndim],
            dtype=np.uint32,
        )
        if track_permutation_indices
        else None
    )
    permutation_flips = (
        np.empty(
            [training_steps, layers - 1, ndim],
            dtype=bool,
        )
        if track_permutation_flips
        else None
    )
    opt_params_hist = (
        [None for _ in range(training_steps)] if track_opt_params_hist else None
    )
    w_hist = (
        np.empty(
            [training_steps, layers, ndim, ndim],
            dtype=np.float32,
        )
        if track_w_hist
        else None
    )
    b_hist = (
        np.empty(
            [training_steps, layers, ndim],
            dtype=np.float32,
        )
        if track_b_hist
        else None
    )
    w_ideal_hist = (
        np.empty(
            [training_steps, layers, ndim, ndim],
            dtype=np.float32,
        )
        if track_w_ideal_hist
        else None
    )
    b_ideal_hist = (
        np.empty(
            [training_steps, layers, ndim],
            dtype=np.float32,
        )
        if track_b_ideal_hist
        else None
    )

    if verbose:
        print(prefix + "Entering the training loop...")

    # Training loop:
    train_one_percent = training_steps // 100
    assert (
        train_one_percent * 100 == training_steps
    ), f"Training steps must be a multiple of 100, but it was {training_steps}"
    if verbose:
        t0 = time()
    for pct in range(100):
        for i in range(pct * train_one_percent, (pct + 1) * train_one_percent):

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
                losses[i] = jnp.array(L)

            for layer in range(layers):

                if weight_distances is not None:
                    wb_actual = wb(w)
                    wb_ideal = wb(w_ideal)
                    weight_distances[i, layer] = np.sum(
                        np.abs(wb_actual[layer] - wb_ideal[layer])
                    )

                if permutation_indices is not None:
                    permutation_indices[i] = (
                        np.array([p.indices for p in permutation])
                        if layers > 1
                        else np.empty([0, 1], dtype=np.uint32)
                    )

                if permutation_flips is not None:
                    permutation_flips[i] = (
                        np.array([p.flip for p in permutation])
                        if layers > 1
                        else np.empty([0, 1], dtype=bool)
                    )

            if opt_params_hist is not None:
                opt_params_hist[i] = opt_params

            if w_hist is not None:
                w_hist[i] = w.W

            if b_hist is not None:
                b_hist[i] = w.B

            if w_ideal_hist is not None:
                w_ideal_hist[i] = w_ideal.W

            if b_ideal_hist is not None:
                b_ideal_hist[i] = w_ideal.B

        if verbose:
            print(prefix + f"{pct + 1}%")
    if verbose:
        print(prefix + f"Done in {int(time() - t0)} seconds")

    # return (
    #     losses,
    #     weight_distances,
    #     permutation_indices,
    #     permutation_flips,
    #     opt_params_hist,
    #     w_hist,
    #     b_hist,
    #     w_ideal_hist,
    #     b_ideal_hist,
    # )

    if verbose:
        print(prefix + "Saving...")

    cwd = os.getcwd()
    cwd = os.path.join(cwd, *subdir)
    path = lambda *args: os.path.join(cwd, *args)

    if os.path.exists(path()):
        shutil.rmtree(path(), ignore_errors=True)
    os.makedirs(path("weight_distances"))
    os.makedirs(path("optimizer"))
    os.makedirs(path("permutations"))
    os.makedirs(path("weights"))

    w_ideal_permuted = permutations.permute_hidden_layers(w_ideal, permutation)

    @jaxtyped(typechecker=beartype)
    def save(x, *args) -> None:
        if verbose:
            print(prefix + f"Saving `{path(*args)}`...")
        np.save(path(*args), np.array(x), allow_pickle=False)

    if losses is not None:
        save(losses, "losses.npy")

    for i in range(layers):

        save(
            weight_distances[:, i],
            "weight_distances",
            f"layer_{i}.npy",
        )

        if track_convergence:
            save(
                jnp.less(
                    jnp.sum(jnp.array(weight_distances[-1])),
                    jnp.sum(jnp.array(weight_distances[0])),
                ),
                "closer_or_not.npy",
            )

        os.makedirs(path("weights", f"layer_{i}", "weights"))
        os.makedirs(path("weights", f"layer_{i}", "biases"))
        save(
            jnp.array([jnp.ravel(wh[i]) for wh in w_hist]),
            "weights",
            f"layer_{i}",
            "weights",
            "w_historical.npy",
        )
        save(
            jnp.array([jnp.ravel(bh[i]) for bh in b_hist]),
            "weights",
            f"layer_{i}",
            "biases",
            "w_historical.npy",
        )
        save(
            jnp.array([jnp.ravel(wh[i]) for wh in w_ideal_hist]),
            "weights",
            f"layer_{i}",
            "weights",
            "w_ideal_historical.npy",
        )
        save(
            jnp.array([jnp.ravel(bh[i]) for bh in b_ideal_hist]),
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

    for i in range(layers - 1):
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
                [jnn.sigmoid(p.inv_sig_moving_average_decay) for p in opt_params_hist]
            ),
            "optimizer",
            "moving_average_decay.npy",
        )
    if hasattr(opt_params_hist[0], "inv_sig_moving_square_decay"):
        save(
            jnp.array(
                [jnn.sigmoid(p.inv_sig_moving_square_decay) for p in opt_params_hist]
            ),
            "optimizer",
            "moving_square_decay.npy",
        )
    if hasattr(opt_params_hist[0], "inv_sig_moving_square_quotient"):
        save(
            jnp.array(
                [jnn.sigmoid(p.inv_sig_moving_square_quotient) for p in opt_params_hist]
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
            jnp.array([jnn.sigmoid(p.inv_sig_weight_decay) for p in opt_params_hist]),
            "optimizer",
            "weight_decay.npy",
        )
    if hasattr(opt_params_hist[0], "log_epsilon"):
        save(
            jnp.array([jnp.exp(p.log_epsilon) for p in opt_params_hist]),
            "optimizer",
            "epsilon.npy",
        )
