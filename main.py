print("Importing libraries...")


import plot  # Relative import: `plot.py`

from metaoptimizer import feedforward, trial

from jax import nn as jnn, numpy as jnp, random as jrnd


print("Setting hyperparameters...")


# Hyper-hyperparameters?
NDIM = 3  # Input/output vector dimensionality
BATCH = 1  # Number of inputs tp propagate in parallel
LAYERS = 2  # Number of `nl(W @ x + B)` layers in our feedforward model
NONLINEARITY = jnn.gelu
POWER = jnp.array(2.0, dtype=jnp.float32)  # e.g. 1 for L1 loss, 2 for L2, etc.
# TODO: make `POWER` learnable
TRAINING_STEPS = 10000
INITIAL_DISTANCE = jnp.array(0.5, dtype=jnp.float64)
from metaoptimizer.optimizers import (
    adam as optim,
    # sgd as optim,
    # swiss_army_knife as optim,
)

forward_pass = lambda weights, x: feedforward.run(weights, x, NONLINEARITY)
shapes = tuple([NDIM for _ in range(LAYERS + 1)])
w_example = feedforward.init(shapes, jrnd.PRNGKey(42), False)
optimizer = optim.update
opt_params = optim.defaults()
opt_state = optim.init(w_example, opt_params)


print("Running...")


# this prints its own messages:
trial.run(
    jrnd.PRNGKey(42),
    NDIM,
    BATCH,
    LAYERS,
    forward_pass,
    optimizer,
    opt_state,
    opt_params,
    NONLINEARITY,
    TRAINING_STEPS,
    ("logs",),
    POWER,
    INITIAL_DISTANCE,
    "",
)


print("Plotting...")


plot.run("logs")


print("Done")
