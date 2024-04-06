print("Importing...")

from metaoptimizer import feedforward, training
from metaoptimizer.optimizers import swiss_army_knife as optim

from jax import jit, nn as jnn, numpy as jnp, random as jrnd
from jax.experimental.checkify import all_checks, checkify
import numpy as np
import os


print("Setup...")


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
opt_params = optim.defaults()
opt_state = optim.init(w, opt_params)

# Replicable pseudorandomness
key = jrnd.PRNGKey(42)  # the answer

# Network configuration (right now, just choosing the nonlinearity)
forward_pass = feedforward.run
jit_forward_pass = jit(checkify(forward_pass, errors=all_checks))

# Record-keeping
losses = np.empty([EPOCHS], dtype=np.float32)
NONTERMINAL_LAYERS = LAYERS - 1
permutation_indices = [
    np.empty([EPOCHS, NDIM], dtype=np.uint32) for _ in range(NONTERMINAL_LAYERS)
]
permutation_flips = [
    np.empty([EPOCHS, NDIM], dtype=bool) for _ in range(NONTERMINAL_LAYERS)
]


print("Entering training loop...")


# Training loop:
for i in range(EPOCHS):
    k, key = jrnd.split(key)
    x = jrnd.normal(k, [BATCH, NDIM])
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
    assert losses[i].shape == L.shape, f"{losses[i].shape} == {L.shape}"
    losses[i] = np.array(L)
    assert (
        len(permutation_indices)
        == len(permutation_flips)
        == len(permutation)
        == NONTERMINAL_LAYERS
    ), f"{len(permutation_indices)} == {len(permutation_flips)} == {len(permutation)} == {NONTERMINAL_LAYERS}"
    for j in range(NONTERMINAL_LAYERS):
        assert (
            permutation_indices[j][i].shape == permutation[j].indices.shape
        ), f"{permutation_indices[j][i].shape} == {permutation[j].indices.shape}"
        permutation_indices[j][i] = np.array(permutation[j].indices)
        assert (
            permutation_flips[j][i].shape == permutation[j].flip.shape
        ), f"{permutation_flips[j][i].shape} == {permutation[j].flip.shape}"
        permutation_flips[j][i] = np.array(permutation[j].flip)
    print(f"{i}/{EPOCHS}")


print("Saving...")


os.makedirs("logs/permutations", exist_ok=True)
np.save("logs/losses.npy", losses, allow_pickle=False)
for i in range(NONTERMINAL_LAYERS):
    np.save(
        "logs/permutations/layer_{i}_indices.npy",
        permutation_indices[i],
        allow_pickle=False,
    )
    np.save(
        "logs/permutations/layer_{i}_flips.npy",
        permutation_flips[i],
        allow_pickle=False,
    )


print("Done")
