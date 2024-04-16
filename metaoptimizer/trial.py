from metaoptimizer import feedforward, permutations, training
from metaoptimizer.feedforward import Weights
from metaoptimizer.permutations import Permutation
from metaoptimizer.training import ForwardPass, Optimizer

from beartype import beartype
from beartype.typing import Callable, List, Tuple
from functools import partial
from importlib import import_module
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax.experimental import io_callback
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jax_dataclasses import Static
from jaxtyping import jaxtyped, Array, Float32, Float64, PyTree
import matplotlib.pyplot as plt
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
        print("RUNNING `jit_step` FOR THE FIRST TIME")
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
        Callable[
            [Float32[Array, "*n"]],
            Float32[Array, "*n"],
        ]
    ] = jnn.gelu,
    optim: Static[ModuleType] = import_module("metaoptimizer.optimizers.sgd"),
    training_steps: Static[int] = 100000,
    subdir: Static[List[str]] = ["logs"],
    convergence_only: Static[bool] = False,
    power: Float32[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
    lr: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
    initial_distance: Float64[Array, ""] = jnp.array(0.1, dtype=jnp.float64),
    prefix: str = "",
) -> None:

    print(prefix + "Setting up the model architecture...")

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

    # Record-keeping
    weight_distances = [
        [jnp.empty([], dtype=jnp.float32) for _ in range(training_steps)]
        for _ in range(layers)
    ]
    nonterminal_layers = layers - 1
    if not convergence_only:
        losses = jnp.empty([training_steps], dtype=jnp.float32)
        permutation_indices = [
            jnp.empty([training_steps, ndim], dtype=jnp.uint32)
            for _ in range(nonterminal_layers)
        ]
        permutation_flips = [
            jnp.empty([training_steps, ndim], dtype=bool)
            for _ in range(nonterminal_layers)
        ]
        opt_params_hist: List[PyTree[Float64[Array, ""]]] = []
        w_hist: List[PyTree[Float64[Array, "..."]]] = []
        w_ideal_hist: List[PyTree[Float64[Array, "..."]]] = []

    print(prefix + "Compiling the training loop...")

    # Training loop:
    t0 = time()
    EPOCH_PERCENT = training_steps // 100
    assert training_steps == EPOCH_PERCENT * 100  # i.e. divisible by 100
    for percent in range(100):
        for i in range(EPOCH_PERCENT * percent, EPOCH_PERCENT * (percent + 1)):
            key, w, opt_state, opt_params, permutation, L = step(
                batch,
                ndim,
                partial(feedforward.run, nl=nonlinearity),
                optimizer,
                w_ideal,
                opt_state,
                opt_params,
                w,
                power,
                key,
            )

            for j in range(layers):
                dist = jnp.array(
                    jnp.sum(jnp.abs(w.W[j] - w_ideal.B[j]))
                    + jnp.sum(jnp.abs(w.B[j] - w_ideal.B[j]))
                )
                weight_distances[j][i] = dist

            if not convergence_only:
                losses[i] = jnp.array(L)
                for j in range(nonterminal_layers):
                    permutation_indices[j][i] = jnp.array(permutation[j].indices)
                    permutation_flips[j][i] = jnp.array(permutation[j].flip)
                opt_params_hist.append(opt_params)
                w_hist.append(tree_map(jnp.ravel, w))
                w_ideal_hist.append(
                    tree_map(
                        jnp.ravel,
                        permutations.permute_hidden_layers(w_ideal, permutation),
                    )
                )

        print(prefix + f"{percent + 1}%")

    print(prefix + f"Done in {int(time() - t0)} seconds")
    print(prefix + "Saving...")

    w_ideal_permuted = permutations.permute_hidden_layers(w_ideal, permutation)

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
