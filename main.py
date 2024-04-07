print("Importing libraries...")

import plot  # Relative import: `plot.py`

from metaoptimizer import feedforward, training
from metaoptimizer.optimizers import swiss_army_knife as optim

from jax import jit, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental.checkify import all_checks, checkify
from jax.tree_util import tree_flatten, tree_map
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import shutil
from time import time


print("Setting up the model architecture...")


# Hyper-hyperparameters?
NDIM = 3  # Input/output vector dimensionality
BATCH = 32  # Number of inputs tp propagate in parallel
LAYERS = 3  # Number of `nl(W @ x + B)` layers in our feedforward model
POWER = jnp.array(1.0)  # e.g. 1 for L1 loss, 2 for L2, etc.
EPOCHS = 10000

# Weight initialization (note `w_ideal` is really the *goal*)
[w, w_ideal] = [
    feedforward.init(
        [NDIM for _ in range(LAYERS + 1)],
        jrnd.PRNGKey(42 + i),
    )
    for i in range(2)
]

# Optimizer initialization
# TODO: Is there a Pythonic way to reduce redundancy here?
optimizer = optim.update
opt_params = optim.defaults(
    momentum=jnp.array(0.9),
    overstep=jnp.array(0.001),
)
opt_state = optim.init(w, opt_params)

# Replicable pseudorandomness
key = jrnd.PRNGKey(42)  # the answer

# Network configuration (right now, just choosing the nonlinearity)
forward_pass = feedforward.run
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


print("JIT-compiling the training loop...")


# Training loop:
t0 = time()
EPOCH_PERCENT = EPOCHS // 100
assert EPOCHS == EPOCH_PERCENT * 100  # i.e. divisible by 100
for century in range(0, EPOCHS, EPOCH_PERCENT):
    for i in range(century, century + EPOCH_PERCENT):
        k, key = jrnd.split(key)
        x = jrnd.normal(k, [BATCH, NDIM])
        err, y_ideal = jit_forward_pass(w_ideal, x)
        err.throw()
        # err, (w, opt_state, L) = training.step(
        #     w,
        #     forward_pass,
        #     x,
        #     y_ideal,
        #     optimizer,
        #     opt_params,
        #     opt_state,
        #     POWER,
        # )

        # TODO: RENAME `_` to `opt_params` to re-enable learning optimizer params!
        err, (w, opt_state, _, permutation, L) = training.step_global(
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
    print(f"{(century // EPOCH_PERCENT) + 1}%")


print(f"Done in {int(time() - t0)} seconds")
print("Saving...")


cwd = os.getcwd()
path = lambda *args: os.path.join(cwd, *args)
if os.path.exists(path("logs")):
    shutil.rmtree(path("logs"), ignore_errors=True)
os.makedirs(path("logs", "optimizer"), exist_ok=True)
os.makedirs(path("logs", "permutations"), exist_ok=True)
os.makedirs(path("logs", "weight_distances"), exist_ok=True)
np.save(path("logs", "losses.npy"), losses, allow_pickle=False)
for i in range(LAYERS):
    np.save(
        path("logs", "weight_distances", f"layer_{i}.npy"),
        weight_distances[i],
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
np.save(
    path("logs", "optimizer", "lr.npy"),
    np.array([jnp.exp(p.log_lr) for p in opt_params_hist]),
)
np.save(
    path("logs", "optimizer", "moving_average_decay.npy"),
    np.array([jnn.sigmoid(p.inv_sig_moving_average_decay) for p in opt_params_hist]),
)
np.save(
    path("logs", "optimizer", "moving_square_decay.npy"),
    np.array([jnn.sigmoid(p.inv_sig_moving_square_decay) for p in opt_params_hist]),
)
np.save(
    path("logs", "optimizer", "momentum.npy"),
    np.array([jnn.sigmoid(p.inv_sig_momentum) for p in opt_params_hist]),
)
np.save(
    path("logs", "optimizer", "overstep.npy"),
    np.array([jnp.exp(p.log_overstep) for p in opt_params_hist]),
)
np.save(
    path("logs", "optimizer", "weight_decay.npy"),
    np.array([jnn.sigmoid(p.inv_sig_weight_decay) for p in opt_params_hist]),
)
np.save(
    path("logs", "optimizer", "epsilon.npy"),
    np.array([jnp.exp(p.log_epsilon) for p in opt_params_hist]),
)


print("Plotting...")


plot.run()


print("Done")
