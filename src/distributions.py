from functools import partial
from jax import Array, jit, numpy as jnp, scipy as jsp
from jax.numpy import linalg as jla
from jax.lax import cond


@partial(jit, static_argnames=["axis"])
def normalize(x: Array, axis=None) -> Array:
    """
    Normalize a distribution s.t.
    each row (i.e., each index along the last axis)
    has zero mean and unit variance.
    """
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    return (x - mean) / std


@jit
def kabsch(to_be_rotated: Array, target: Array) -> Array:
    """
    Kabsch's algorithm for rotating pairs of points to minimize post-rotation distance.
    Note that we do NOT perform the following steps of Kabsch's original algorithm:
    - Instead of centering both point-clouds at the origin, we leave them at their original positions,
      since points' relationship to the origin is meaningful.
    - The final "rotation matrix" need not be a (right-handed) rotation matrix:
      if axes are negated outside a usual rotation, this is fine,
      as long as they are consistently negated throughout (later, outside this function).
    Example use:
    ```python
    R = kabsch(to_be_rotated, target)
    rotated = to_be_rotated @ R  # should be arbitrarily close to `target`, save roundoff error
    ```
    """
    assert (
        to_be_rotated.ndim == 3
    ), "Please send 3-axis tensors to `kabsch`: (batch, points, values)"
    assert to_be_rotated.shape == target.shape
    covariance = to_be_rotated.transpose(0, 2, 1) @ target
    u, _, vT = jla.svd(covariance)
    return u @ vT
