from metaoptimizer import feedforward, permutations, training
from metaoptimizer.feedforward import Weights
from metaoptimizer.permutations import Permutation
from metaoptimizer.training import ForwardPass, Optimizer

from beartype import beartype
from beartype.typing import Callable, List, Tuple
from functools import partial
from importlib import import_module
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jax_dataclasses import Static
from jaxtyping import jaxtyped, Array, Float32, Float64, PyTree
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import shutil
from time import time
from types import ModuleType


if os.getenv("NONJIT") == "1":
    print("NOTE: `NONJIT` activated")
else:
    from jax.experimental.checkify import all_checks, checkify, Error
    from jax_dataclasses import jit


@jaxtyped(typechecker=beartype)
def raw_step(
    batch: Static[int],
    ndim: Static[int],
    forward_pass: Static[ForwardPass],
    optimizer: Static[Optimizer],
    w_ideal: Weights,
    opt_state: PyTree[Float64[Array, "..."]],
    opt_params: PyTree[Float64[Array, ""]],
    w: Weights,
    power: Float32[Array, ""],
    key: Array,
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


if os.getenv("NONJIT") == "1":

    def step(
        batch: Static[int],
        ndim: Static[int],
        forward_pass: Static[ForwardPass],
        optimizer: Static[Optimizer],
        w_ideal: Weights,
        opt_state: PyTree[Float64[Array, "..."]],
        opt_params: PyTree[Float64[Array, ""]],
        w: Weights,
        power: Float32[Array, ""],
        key: Array,
    ) -> Tuple[
        Array,
        Weights,
        PyTree[Float64[Array, "..."]],
        PyTree[Float64[Array, ""]],
        List[Permutation],
        Float32[Array, ""],
    ]:
        return raw_step(
            batch,
            ndim,
            forward_pass,
            optimizer,
            w_ideal,
            opt_state,
            opt_params,
            w,
            power,
            key,
        )

else:

    @jit
    def jit_step(
        batch: Static[int],
        ndim: Static[int],
        forward_pass: Static[ForwardPass],
        optimizer: Static[Optimizer],
        w_ideal: Weights,
        opt_state: PyTree[Float64[Array, "..."]],
        opt_params: PyTree[Float64[Array, ""]],
        w: Weights,
        power: Float32[Array, ""],
        key: Array,
    ) -> Tuple[
        Error,
        Tuple[
            Array,
            Weights,
            PyTree[Float64[Array, "..."]],
            PyTree[Float64[Array, ""]],
            List[Permutation],
            Float32[Array, ""],
        ],
    ]:
        return checkify(raw_step, errors=all_checks)(
            batch,
            ndim,
            forward_pass,
            optimizer,
            w_ideal,
            opt_state,
            opt_params,
            w,
            power,
            key,
        )

    def step(
        batch: Static[int],
        ndim: Static[int],
        forward_pass: Static[ForwardPass],
        optimizer: Static[Optimizer],
        w_ideal: Weights,
        opt_state: PyTree[Float64[Array, "..."]],
        opt_params: PyTree[Float64[Array, ""]],
        w: Weights,
        power: Float32[Array, ""],
        key: Array,
    ) -> Tuple[
        Array,
        Weights,
        PyTree[Float64[Array, "..."]],
        PyTree[Float64[Array, ""]],
        List[Permutation],
        Float32[Array, ""],
    ]:
        err, y = jit_step(
            batch,
            ndim,
            forward_pass,
            optimizer,
            w_ideal,
            opt_state,
            opt_params,
            w,
            power,
            key,
        )
        err.throw()
        return y


@jaxtyped(typechecker=beartype)
def run(
    key: Array,
    ndim: Static[int] = 3,
    batch: Static[int] = 1,
    layers: Static[int] = 2,
    nonlinearity: Static[
        Callable[[Float32[Array, "*n"]], Float32[Array, "*n"]]
    ] = jnn.gelu,
    optim: Static[ModuleType] = import_module("metaoptimizer.optimizers.sgd"),
    training_steps: Static[int] = 100000,
    subdir: Static[List[str]] = [],
    convergence_only: Static[bool] = False,
    power: Float32[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
    lr: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
    initial_distance: Float64[Array, ""] = jnp.array(0.1, dtype=jnp.float64),
) -> None:

    print("Setting up the model architecture...")

    k1, k2 = jrnd.split(key)

    # Weight initialization (note `w_ideal` is really the *goal*)
    w_ideal = feedforward.init(
        [ndim for _ in range(layers + 1)],
        k1,
        random_biases=True,
    )

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

    # Replicable pseudorandomness
    key = jrnd.PRNGKey(42)  # the answer

    # Network configuration (right now, just choosing the nonlinearity)
    forward_pass = partial(feedforward.run, nl=nonlinearity)

    # Record-keeping
    weight_distances = [
        np.empty([training_steps], dtype=np.float32) for _ in range(layers)
    ]
    if not convergence_only:
        nonterminal_layers = layers - 1
        losses = np.empty([training_steps], dtype=np.float32)
        permutation_indices = [
            np.empty([training_steps, ndim], dtype=np.uint32)
            for _ in range(nonterminal_layers)
        ]
        permutation_flips = [
            np.empty([training_steps, ndim], dtype=bool)
            for _ in range(nonterminal_layers)
        ]
        opt_params_hist: List[PyTree[Float64[Array, ""]]] = []
        w_hist: List[PyTree[Float64[Array, "..."]]] = []
        w_ideal_hist: List[PyTree[Float64[Array, "..."]]] = []

    print("Compiling the training loop...")

    # Training loop:
    t0 = time()
    EPOCH_PERCENT = training_steps // 100
    assert training_steps == EPOCH_PERCENT * 100  # i.e. divisible by 100
    for percent in range(100):
        for i in range(EPOCH_PERCENT * percent, EPOCH_PERCENT * (percent + 1)):
            key, w, opt_state, opt_params, permutation, L = step(
                batch,
                ndim,
                forward_pass,
                optimizer,
                w_ideal,
                opt_state,
                opt_params,
                w,
                power,
                key,
            )

            for j in range(layers):
                dist = np.array(
                    jnp.sum(jnp.abs(w.W[j] - w_ideal.B[j]))
                    + jnp.sum(jnp.abs(w.B[j] - w_ideal.B[j]))
                )
                weight_distances[j][i] = dist

            if not convergence_only:
                losses[i] = np.array(L)
                for j in range(nonterminal_layers):
                    permutation_indices[j][i] = np.array(permutation[j].indices)
                    permutation_flips[j][i] = np.array(permutation[j].flip)
                opt_params_hist.append(opt_params)
                w_hist.append(tree_map(jnp.ravel, w))
                w_ideal_hist.append(
                    tree_map(
                        jnp.ravel,
                        permutations.permute_hidden_layers(w_ideal, permutation),
                    )
                )

        print(f"{percent + 1}%")

    print(f"Done in {int(time() - t0)} seconds")
    print("Saving...")

    w_ideal_permuted = permutations.permute_hidden_layers(w_ideal, permutation)

    cwd = os.getcwd()
    cwd = os.path.join(cwd, *subdir)
    path = lambda *args: os.path.join(cwd, *args)
    if os.path.exists(path("logs")):
        shutil.rmtree(path("logs"), ignore_errors=True)
    os.makedirs(path("logs", "optimizer"))
    os.makedirs(path("logs", "permutations"))
    os.makedirs(path("logs", "weights"))
    os.makedirs(path("logs", "weight_distances"))
    np.save(path("logs", "losses.npy"), losses, allow_pickle=False)
    for i in range(layers):
        os.makedirs(path("logs", "weights", f"layer_{i}", "weights"))
        os.makedirs(path("logs", "weights", f"layer_{i}", "biases"))
        np.save(
            path("logs", "weights", f"layer_{i}", "weights", "w_historical.npy"),
            np.array([jnp.ravel(wh.W[i]) for wh in w_hist]),
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "biases", "w_historical.npy"),
            np.array([jnp.ravel(wh.B[i]) for wh in w_hist]),
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "weights", "w_ideal_historical.npy"),
            np.array([jnp.ravel(wh.W[i]) for wh in w_ideal_hist]),
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "biases", "w_ideal_historical.npy"),
            np.array([jnp.ravel(wh.B[i]) for wh in w_ideal_hist]),
        )
        np.save(
            path("logs", "weight_distances", f"layer_{i}.npy"),
            weight_distances[i],
            allow_pickle=False,
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "weights", "ideal_orig.npy"),
            w_ideal.W[i],
            allow_pickle=False,
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "weights", "ideal_perm.npy"),
            w_ideal_permuted.W[i],
            allow_pickle=False,
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "weights", "final.npy"),
            w.W[i],
            allow_pickle=False,
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "biases", "ideal_orig.npy"),
            w_ideal.B[i],
            allow_pickle=False,
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "biases", "ideal_perm.npy"),
            w_ideal_permuted.B[i],
            allow_pickle=False,
        )
        np.save(
            path("logs", "weights", f"layer_{i}", "biases", "final.npy"),
            w.B[i],
            allow_pickle=False,
        )
    for i in range(nonterminal_layers):
        np.save(
            path("logs", "permutations", f"layer_{i}_indices.npy"),
            permutation_indices[i],
            allow_pickle=False,
        )
        np.save(
            path("logs", "permutations", f"layer_{i}_flips.npy"),
            permutation_flips[i],
            allow_pickle=False,
        )
    if hasattr(opt_params_hist[0], "log_lr"):
        np.save(
            path("logs", "optimizer", "lr.npy"),
            np.array([jnp.exp(p.log_lr) for p in opt_params_hist]),
        )
    if hasattr(opt_params_hist[0], "inv_sig_moving_average_decay"):
        np.save(
            path("logs", "optimizer", "moving_average_decay.npy"),
            np.array(
                [jnn.sigmoid(p.inv_sig_moving_average_decay) for p in opt_params_hist]
            ),
        )
    if hasattr(opt_params_hist[0], "inv_sig_moving_square_decay"):
        np.save(
            path("logs", "optimizer", "moving_square_decay.npy"),
            np.array(
                [jnn.sigmoid(p.inv_sig_moving_square_decay) for p in opt_params_hist]
            ),
        )
    if hasattr(opt_params_hist[0], "inv_sig_moving_square_quotient"):
        np.save(
            path("logs", "optimizer", "moving_square_quotient.npy"),
            np.array(
                [jnn.sigmoid(p.inv_sig_moving_square_quotient) for p in opt_params_hist]
            ),
        )
    if hasattr(opt_params_hist[0], "inv_sig_momentum"):
        np.save(
            path("logs", "optimizer", "momentum.npy"),
            np.array([jnn.sigmoid(p.inv_sig_momentum) for p in opt_params_hist]),
        )
    if hasattr(opt_params_hist[0], "log_overstep"):
        np.save(
            path("logs", "optimizer", "overstep.npy"),
            np.array([jnp.exp(p.log_overstep) for p in opt_params_hist]),
        )
    if hasattr(opt_params_hist[0], "inv_sig_weight_decay"):
        np.save(
            path("logs", "optimizer", "weight_decay.npy"),
            np.array([jnn.sigmoid(p.inv_sig_weight_decay) for p in opt_params_hist]),
        )
    if hasattr(opt_params_hist[0], "log_epsilon"):
        np.save(
            path("logs", "optimizer", "epsilon.npy"),
            np.array([jnp.exp(p.log_epsilon) for p in opt_params_hist]),
        )
