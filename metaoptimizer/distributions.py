from beartype import beartype
from beartype.typing import Tuple
from jax import numpy as jnp, scipy as jsp
from jax.experimental.checkify import check
from jax.numpy import linalg as jla
from jax.lax import cond
from jaxtyping import jaxtyped, Array, Float


@jaxtyped(typechecker=beartype)
def normalize(x: Float[Array, "..."], axis=None) -> Array:
    """
    Normalize a distribution s.t.
    each row (i.e., each index along the last axis)
    has zero mean and unit variance.
    """
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + 1e-8)


@jaxtyped(typechecker=beartype)
def kabsch(
    to_be_rotated: Float[Array, "batch points ndim"],
    target: Float[Array, "batch points ndim"],
) -> Float[Array, "batch ndim ndim"]:
    """
    Kabsch's algorithm for (possibly improperly) rotating pairs of points to minimize post-rotation distance.
    Note that we do NOT perform the following steps of Kabsch's original algorithm:
    - Instead of centering both point-clouds at the origin, we leave them at their original positions,
      since points' relationship to the origin is meaningful.
    - The final "rotation matrix" need not be a (right-handed) rotation matrix:
      if ndim are negated outside a usual rotation, this is fine,
      as long as they are consistently negated throughout (later, outside this function).
    Example use:
    ```python
    R = kabsch(to_be_rotated, target)
    rotated = to_be_rotated @ R  # should be arbitrarily close to `target`, save roundoff error
    ```
    """
    covariance = to_be_rotated.transpose(0, 2, 1) @ target
    u, _, vT = jla.svd(covariance)
    return u @ vT


@jaxtyped(typechecker=beartype)
def rotate_and_compare(
    actual: Float[Array, "batch points ndim"],
    ideal: Float[Array, "batch points ndim"],
) -> Tuple[
    Float[Array, ""],
    Float[Array, "batch points ndim"],
    Float[Array, "batch ndim ndim"],
]:
    """
    Calculate a (possibly improper) rotation matrix to get `actual` as close to `ideal` as possible
    (namely, the `R` that minimizes the norm of `(actual @ R) - ideal`)
    and return the norm we just minimized.
    Returns `(norm, ideal @ R, R)`
    """
    R = kabsch(actual, ideal)
    aR = actual @ R
    return jla.norm(aR - ideal), aR, R


@jaxtyped(typechecker=beartype)
def inverse_sigmoid(x: Float[Array, "*n"]) -> Float[Array, "*n"]:
    # https://stackoverflow.com/questions/10097891/inverse-logistic-sigmoid-function
    check(jnp.all(0 < x) and jnp.all(x < 1), "{x} must be between 0 and 1", x=x)
    return jnp.log(x / (1 - x))
