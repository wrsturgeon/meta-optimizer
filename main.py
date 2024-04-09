print("Importing libraries...")

import plot  # Relative import: `plot.py`

from metaoptimizer import feedforward, permutations, training

from functools import partial
from jax import jit, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental.checkify import all_checks, checkify
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import shutil
from time import time


print("Setting up the model architecture...")


# Hyper-hyperparameters?
NDIM = 3  # Input/output vector dimensionality
BATCH = 1  # Number of inputs tp propagate in parallel
LAYERS = 2  # Number of `nl(W @ x + B)` layers in our feedforward model
NONLINEARITY = jnn.gelu
POWER = jnp.array(2.0, dtype=jnp.float32)  # e.g. 1 for L1 loss, 2 for L2, etc.
# TODO: make `POWER` learnable
LR = jnp.array(0.001, dtype=jnp.float64)
EPOCHS = 10000
from metaoptimizer.optimizers import (
    swiss_army_knife as optim,
    # sgd as optim,
)

# Weight initialization (note `w_ideal` is really the *goal*)
w = feedforward.init(
    [NDIM for _ in range(LAYERS + 1)],
    jrnd.PRNGKey(42),
    random_biases=False,
)
w_ideal = feedforward.init(
    [NDIM for _ in range(LAYERS + 1)],
    jrnd.PRNGKey(43),
    random_biases=True,
)

# Uncomment if you want `w` to start already very close to `w_ideal`:
std_distance = 1.0
w_flat, w_def = tree_flatten(w)
w_keys = tree_unflatten(w_def, jrnd.split(jrnd.PRNGKey(42), len(w_flat)))
w = tree_map(lambda x, k: x + std_distance * jrnd.normal(k, x.shape), w_ideal, w_keys)

# Optimizer initialization
# TODO: Is there a Pythonic way to reduce redundancy here?
optimizer = optim.update
opt_params = optim.defaults(lr=LR)
opt_state = optim.init(w, opt_params)

# Replicable pseudorandomness
key = jrnd.PRNGKey(42)  # the answer

# Network configuration (right now, just choosing the nonlinearity)
forward_pass = partial(feedforward.run, nl=NONLINEARITY)
jit_forward_pass = jit(checkify(forward_pass, errors=all_checks))

# Record-keeping
NONTERMINAL_LAYERS = LAYERS - 1
losses = np.empty([EPOCHS], dtype=np.float32)
weight_distances = [np.empty([EPOCHS], dtype=np.float32) for _ in range(LAYERS)]
permutation_indices = [
    np.empty([EPOCHS, NDIM], dtype=np.uint32) for _ in range(NONTERMINAL_LAYERS)
]
permutation_flips = [
    np.empty([EPOCHS, NDIM], dtype=bool) for _ in range(NONTERMINAL_LAYERS)
]
opt_params_hist = []
w_hist = []
w_ideal_hist = []


print("Compiling the training loop...")


# Training loop:
t0 = time()
EPOCH_PERCENT = EPOCHS // 100
assert EPOCHS == EPOCH_PERCENT * 100  # i.e. divisible by 100
for percent in range(100):
    for i in range(EPOCH_PERCENT * percent, EPOCH_PERCENT * (percent + 1)):
        k, key = jrnd.split(key)
        x = jrnd.normal(k, [BATCH, NDIM], dtype=jnp.float32)
        err, y_ideal = jit_forward_pass(w_ideal, x)
        err.throw()
        err, (w, opt_state, opt_params, permutation, L) = training.step_global(
            w,
            forward_pass,
            x,
            y_ideal,
            optimizer,
            opt_params,
            opt_state,
            w_ideal,
            POWER,
        )
        err.throw()
        losses[i] = np.array(L)
        for j in range(LAYERS):
            dist = np.array(
                jnp.sum(jnp.abs(w.W[j] - w_ideal.B[j]))
                + jnp.sum(jnp.abs(w.B[j] - w_ideal.B[j]))
            )
            weight_distances[j][i] = dist
        for j in range(NONTERMINAL_LAYERS):
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
path = lambda *args: os.path.join(cwd, *args)
if os.path.exists(path("logs")):
    shutil.rmtree(path("logs"), ignore_errors=True)
os.makedirs(path("logs", "optimizer"))
os.makedirs(path("logs", "permutations"))
os.makedirs(path("logs", "weights"))
os.makedirs(path("logs", "weight_distances"))
np.save(path("logs", "losses.npy"), losses, allow_pickle=False)
for i in range(LAYERS):
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
for i in range(NONTERMINAL_LAYERS):
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
        np.array([jnn.sigmoid(p.inv_sig_moving_square_decay) for p in opt_params_hist]),
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


print("Plotting...")


plot.run()


print("Done")
