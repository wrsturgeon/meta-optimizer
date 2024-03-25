from metaoptimizer.nontest import jit
from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Callable
from functools import partial
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax.experimental import checkify
from jax.numpy import linalg as jla
from jaxtyping import jaxtyped, Array, Float, UInt

KeyArray = UInt[Array, "n_keys"]  # <https://github.com/google/jax/issues/12706>


@partial(jit, static_argnames=["nl"])
@jaxtyped(typechecker=beartype)
def feedforward(
    w: Weights,
    x: Float[Array, "batch n_in"],
    nl: Callable[[Float[Array, "..."]], Float[Array, "..."]],  # = jnn.gelu,
) -> Float[Array, "batch n_out"]:
    batch, ndim_in = x.shape
    x = x[..., jnp.newaxis]
    checkify.check(
        jnp.all(jnp.isfinite(x)),
        "`feedforward` got an input `x` with non-finite values",
    )
    n = w.layers()
    for i in range(n):
        x = nl(w.W[i] @ x + w.B[i][jnp.newaxis, ..., jnp.newaxis])
        checkify.check(
            jnp.all(jnp.isfinite(x)),
            f"`feedforward` produced a hidden `x` with non-finite values (after layer #{i})",
        )
    return x[..., 0]


# shouldn't be JITted b/c only run once
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
    return Weights(W, B)


@jit
@jaxtyped(typechecker=beartype)
def rotate_weights(
    w: Weights,
    R: list[Float[Array, "..."]],
) -> Weights:
    assert isinstance(R, list)
    n = len(R)
    assert w.layers() == n + 1
    # Note that "permutations" should occur only vertically,
    # since, in `Wx + B`, horizontal indices of `W`
    # track indices of the column-vector `x`.
    # So we're looking for `RW` instead of `WR`.
    for i in range(n):
        w.W[i] = R[i] @ w.W[i]
        w.B[i] = R[i] @ w.B[i]
        RT = R[i].T
        Rinv = jla.inv(R[i])
        inv_diff = jnp.abs(RT - Rinv)
        assert jnp.all(inv_diff < 0.01)
        w.W[-1] = R[i].T @ w.W[-1]
        # B[-1] = R[i].T @ B[-1]
    return w


# TODO: Try using ML to find rotation matrices s.t.
# each rotation matrix minimizes distance to the true weights
# but all matrices, taken together, produce an identity matrix
# (so the output indices really mean what they should,
# instead of a permutation thereof).
# Right now, we're just unilaterally reverse-rotating the last `W`,
# which is probably far from ideal but much faster.
