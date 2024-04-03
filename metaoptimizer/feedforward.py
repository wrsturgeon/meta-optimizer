from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Callable
from functools import partial
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax.experimental.checkify import check
from jax.numpy import linalg as jla
from jaxtyping import jaxtyped, Array, Float, PyTree, UInt

KeyArray = UInt[Array, "n_keys"]  # <https://github.com/google/jax/issues/12706>


@jaxtyped(typechecker=beartype)
def feedforward(
    w: Weights,
    x: Float[Array, "batch n_in"],
    nl: Callable[[Float[Array, "..."]], Float[Array, "..."]],  # = jnn.gelu,
) -> Float[Array, "batch n_out"]:
    batch, ndim_in = x.shape
    x = x[..., jnp.newaxis]
    check(
        jnp.all(jnp.isfinite(x)),
        """
        `feedforward` got an input `x` with non-finite values.
        Original `x` was
        {x}
        """,
        x=x,
    )
    n = w.layers()
    for i in range(n):
        y = nl(w.W[i] @ x + w.B[i][jnp.newaxis, ..., jnp.newaxis])
        check(
            jnp.all(jnp.isfinite(y)),
            """
            `feedforward` produced a hidden `x` with non-finite values (after layer #{i}).
            Original `x` was
            {x}
            `w.W[i]` was
            {w}
            `w.B[i]` was
            {b}
            After, `x` was
            {y}
            """,
            i=jnp.array(i, dtype=jnp.uint32),
            x=x,
            w=w.W[i],
            b=w.B[i],
            y=y,
        )
        x = y
    return x[..., 0]


@jaxtyped(typechecker=beartype)
def feedforward_init(
    sizes: list[int],
    key: KeyArray,
) -> Weights:
    n = len(sizes)
    W = []
    B = []
    init = jnn.initializers.he_normal()
    for i in range(1, n):
        key, k = jrnd.split(key)
        W.append(init(k, (sizes[i], sizes[i - 1]), jnp.float32))
        B.append(jnp.zeros(sizes[i]))
    return Weights(W=W, B=B)
