from functools import partial
from jax import Array, jit, nn as jnn, numpy as jnp, random as jrnd
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


def feedforward_init(
    sizes: list[int],
    key: jrnd.PRNGKey,
) -> tuple[list[Array], list[Array]]:
    n = len(sizes)
    W = []
    B = []
    init = jnn.initializers.he_normal()
    for i in range(1, n):
        key, k = jrnd.split(key)
        W.append(init(k, (sizes[i], sizes[i - 1]), jnp.float32))
        B.append(jnp.zeros(sizes[i]))
    return W, B
