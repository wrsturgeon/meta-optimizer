from beartype import beartype
from beartype.typing import Callable, List, NamedTuple
from check_and_compile import check_and_compile
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float32, Float64


class Weights(NamedTuple):
    W: Float64[Array, "n_layers n_out n_in"]
    B: Float64[Array, "n_layers n_out"]


@jaxtyped(typechecker=beartype)
def layers(w: Weights) -> int:
    assert jnp.issubdtype(w.W.dtype, jnp.float64)
    assert jnp.issubdtype(w.B.dtype, jnp.float64)
    n = w.W.shape[0]
    assert n == w.B.shape[0]
    return n


# @check_and_compile()
@jaxtyped(typechecker=beartype)
def wb(weights: Weights) -> Float32[Array, "n_layers n_out n_in_plus_1"]:
    assert weights.W.ndim == 3
    assert weights.B.ndim == 2
    assert weights.W.shape[:-1] == weights.B.shape
    assert jnp.issubdtype(weights.W.dtype, jnp.float64)
    assert jnp.issubdtype(weights.B.dtype, jnp.float64)
    w = weights.W.astype(jnp.float32)
    b = weights.B.astype(jnp.float32)[..., jnp.newaxis]
    y = jnp.concat([w, b], axis=-1)
    assert y.shape == (*weights.W.shape[:-1], weights.W.shape[-1] + 1)
    return y
