from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Callable, Tuple
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax.numpy import linalg as jla
from jaxtyping import jaxtyped, Array, Float32, Float64, PyTree, UInt16, UInt32
import os


KeyArray = UInt32[Array, "n_keys"]  # <https://github.com/google/jax/issues/12706>


@check_and_compile(3)
def nonlinear(
    x: Float32[Array, "batch n_in"],
    w: Float64[Array, "n_out n_in"],
    b: Float64[Array, "n_out"],
    nl: Callable[[Float32[Array, "batch n_out"]], Float32[Array, "batch n_out"]],
) -> Float32[Array, "batch n_out"]:
    X = x[..., jnp.newaxis]
    W = w.astype(jnp.float32)
    B = b.astype(jnp.float32)[jnp.newaxis, ..., jnp.newaxis]
    Z = W @ X + B
    assert Z.shape[-1] == 1
    z = Z[..., 0]
    return nl(z)


@check_and_compile(2)
def run(
    weights: Weights,
    x: Float32[Array, "batch n_in"],
    nl: Callable[[Float32[Array, "..."]], Float32[Array, "..."]] = jnn.gelu,
) -> Float32[Array, "batch n_out"]:
    for w, b in zip(weights.W, weights.B):
        assert jnp.issubdtype(w, jnp.float64)
        assert jnp.issubdtype(b, jnp.float64)
        x = nonlinear(x, w, b, nl)
    return x


# @check_and_compile(0, 2)
@jaxtyped(typechecker=beartype)
def init(sizes: Tuple, key: KeyArray, random_biases: bool) -> Weights:
    n = len(sizes)
    W = []
    B = []
    he = jnn.initializers.he_normal(dtype=jnp.float64)
    for i in range(1, n):
        size = sizes[i]
        assert isinstance(size, int)
        key, k = jrnd.split(key)
        if random_biases:
            k, k1 = jrnd.split(k)
            B.append(jrnd.normal(k1, [size], dtype=jnp.float64))
        else:
            B.append(jnp.zeros([size], dtype=jnp.float64))
        W.append(he(k, [size, sizes[i - 1]], dtype=jnp.float64))
    return Weights(W=jnp.stack(W), B=jnp.stack(B))
