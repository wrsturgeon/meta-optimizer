from functools import partial
from jax import Array, jit, nn as jnn, numpy as jnp
from typing import Callable


@partial(jit, static_argnames=["nl"])
def feedforward(
    W: list[Array],
    B: list[Array],
    x: Array,
    nl: Callable[Array, Array] = jnn.gelu,
) -> Array:
    n = len(W)
    assert n == len(B)
    for i in range(n):
        x = nl((W[i] @ x) + B[i])
    return x
