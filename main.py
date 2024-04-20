print("Importing libraries...")


import plot  # Relative import: `plot.py`

from jax import nn as jnn, numpy as jnp, random as jrnd
from metaoptimizer import trial


print("Setting hyperparameters...")


# Hyper-hyperparameters?
NDIM = 3  # Input/output vector dimensionality
BATCH = 1  # Number of inputs tp propagate in parallel
LAYERS = 2  # Number of `nl(W @ x + B)` layers in our feedforward model
NONLINEARITY = jnn.gelu
POWER = jnp.array(2.0, dtype=jnp.float32)  # e.g. 1 for L1 loss, 2 for L2, etc.
# TODO: make `POWER` learnable
LR = jnp.array(0.01, dtype=jnp.float64)
TRAINING_STEPS = 100
INITIAL_DISTANCE = jnp.array(0.01, dtype=jnp.float64)
from metaoptimizer.optimizers import (
    swiss_army_knife as optim,
    # sgd as optim,
)


print("Running...")


# this prints its own messages:
trial.run(
    jrnd.PRNGKey(42),
    NDIM,
    BATCH,
    LAYERS,
    NONLINEARITY,
    optim,
    TRAINING_STEPS,
    ("logs",),
    False,
    POWER,
    LR,
    INITIAL_DISTANCE,
    "",
)


print("Plotting...")


plot.run()


print("Done")
