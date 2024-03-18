from functools import partial
from jax import Array, jit, numpy as jnp
from jax.lax import cond


# @partial(jit, static_argnames=["axis"])
def normalize(x: Array, axis=None) -> Array:
    """
    Normalize a distribution s.t.
    each row (i.e., each index along the last axis)
    has zero mean and unit variance.
    """
    mean = jnp.mean(x, axis=axis, keepdims=True)
    print()
    print("Mean:")
    print(mean)
    std = jnp.std(x, axis=axis, keepdims=True)
    print("Standard deviation:")
    print(std)
    return (x - mean) / std
