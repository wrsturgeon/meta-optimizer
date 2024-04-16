print("Importing libraries...")


import plot  # Relative import: `plot.py`

from jax import nn as jnn, numpy as jnp, random as jrnd
from metaoptimizer import trial


# Hyper-hyperparameters?
NDIM = 3  # Input/output vector dimensionality
BATCH = 1  # Number of inputs tp propagate in parallel
LAYERS = 2  # Number of `nl(W @ x + B)` layers in our feedforward model
NONLINEARITY = jnn.gelu
POWER = jnp.array(2.0, dtype=jnp.float32)  # e.g. 1 for L1 loss, 2 for L2, etc.
# TODO: make `POWER` learnable
LR = jnp.array(0.01, dtype=jnp.float64)
TRAINING_STEPS = 100000
from metaoptimizer.optimizers import (
    swiss_army_knife as optim,
    # sgd as optim,
)


# this prints its own messages:
trial.run(
    ndim=NDIM,
    batch=BATCH,
    layers=LAYERS,
    nonlinearity=NONLINEARITY,
    power=POWER,
    lr=LR,
    training_steps=TRAINING_STEPS,
    optim=optim,
    key=jrnd.PRNGKey(42),
)


print("Plotting...")


plot.run()


print("Done")
