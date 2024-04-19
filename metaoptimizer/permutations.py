from metaoptimizer.jit import jit
from metaoptimizer.weights import layers, wb, Weights

from beartype import beartype
from beartype.typing import List, Optional, Tuple
from jax import nn as jnn, numpy as jnp, vmap, ShapeDtypeStruct
from jax.experimental.checkify import check
from jax.lax import cond, fori_loop, stop_gradient
from jax.tree_util import tree_map, tree_reduce
from jaxtyping import (
    jaxtyped,
    Array,
    Bool,
    Float32,
    Float64,
    Int,
    PyTree,
    Shaped,
    UInt16,
)
import operator
from typing import NamedTuple


class Permutation(NamedTuple):
    """Description of a permutation on some tensor (without explicitly carrying around that tensor)."""

    indices: UInt16[Array, "n"]
    flip: Bool[Array, "n"]


@jit(2)
def permute(
    x: Shaped[Array, "*n"],
    permutation: Permutation,
    axis: int,
) -> Shaped[Array, "*n"]:
    n = x.shape[axis]
    assert (
        permutation.indices.shape == permutation.flip.shape == (n,)
    ), f"{permutation.indices.shape} =/= {permutation.flip.shape} =/= ({n},)"
    permuted = jnp.apply_along_axis(lambda z: z[permutation.indices], axis, x)
    flip = permutation.flip
    while flip.ndim < permuted.ndim:
        flip = flip[..., jnp.newaxis]
    return jnp.where(flip, -permuted, permuted)


@jit()
def permute_hidden_layers(
    w: Weights,
    ps: List[Permutation],
) -> Weights:
    """Permute hidden layers' columns locally without changing the output of a network."""
    n = len(ps)
    assert layers(w) == n + 1
    W = w.W
    B = w.B
    for i in range(n):
        p = ps[i]
        W = W.at[i].set(permute(W[i], p, 0))
        W = W.at[i + 1].set(permute(W[i + 1], p, 1))
        B = B.at[i].set(permute(B[i], p, 0))
    return Weights(W=W, B=B)


@jit(2)
def cut_axes(
    x: Shaped[Array, "..."],
    indices: UInt16[Array, "n_indices"],
    axis: int = 0,
) -> Shaped[Array, "n_indices ..."]:
    n = x.shape[axis]
    index_range = jnp.arange(n - 1)
    for _ in range(axis + 1):
        indices = indices[..., jnp.newaxis]
        index_range = index_range[jnp.newaxis]
    for _ in range(axis + 1, x.ndim):
        indices = indices[..., jnp.newaxis]
        index_range = index_range[..., jnp.newaxis]
    cmp = index_range < indices
    left = jnp.apply_along_axis(lambda z: z[:-1], axis, x)[jnp.newaxis]
    right = jnp.apply_along_axis(lambda z: z[1:], axis, x)[jnp.newaxis]
    out = jnp.where(cmp, left, right)

    return out


@jit()
def find_permutation_rec(
    actual: Float32[Array, "n ..."],
    ideal: Float32[Array, "n ..."],
    rowwise: Float32[Array, "n n"],
    flip: Bool[Array, "n n"],
) -> Tuple[Permutation, Float32[Array, ""]]:
    # Note that, in the last two matrices above,
    # the 1st axis represents `_actual`, and
    # the 2nd axis represents `_ideal`, so
    # `rowwise[i, j, k]` is the L1 distance between
    # `actual[i]` and `ideal[j]` (w/ the former flipped iff `k`).
    # Note, further, that we're permuting `actual`,
    # so we should really only ever care about `actual[n]`
    # where `n` is the recursive depth thus far.
    # In other words, we can just delete the top row
    # in each recursive call and not keep track of `n`.

    # # Note: this `assert` (not `check`) will fail only in JIT compilation:
    # assert jnp.all(jnp.isfinite(actual)), "DO NOT JIT! (hours of compilation)"
    # # Problem is aggressive inlining: <https://github.com/google/jax/issues/7155>

    n = actual.shape[0]

    if n == 0:
        return (
            Permutation(
                indices=jnp.array([], dtype=jnp.uint16),
                flip=jnp.array([], dtype=jnp.bool),
            ),
            jnp.array(0, dtype=jnp.float32),
        )

    # TODO: Do we flip here or not?

    # Basic idea is to select the first index of the final permutation,
    # remove the element at that index from the array we'll work with,
    # recurse, increment all indices greater than the one we chose,
    # then `cons` it onto the first index we just chose.
    index_range: UInt16[Array, "n"] = jnp.arange(n, dtype=jnp.uint16)
    recursed, recursed_losses = vmap(find_permutation_rec, in_axes=(0, 0, 0, 0))(
        cut_axes(actual, index_range, 0),
        cut_axes(ideal, index_range, 0),
        cut_axes(rowwise[1:], index_range, 1),
        cut_axes(flip[1:], index_range, 1),
    )

    assert recursed.indices.shape == (
        n,
        n - 1,
    ), f"{recursed.indices.shape} =/= {(n, n - 1)}"

    assert recursed.flip.shape == (
        n,
        n - 1,
    ), f"{recursed.flip.shape} =/= {(n, n - 1)}"

    assert recursed_losses.shape == (n,), f"{recursed_losses.shape} =/= {(n,)}"

    losses: Float32[Array, "n"] = recursed_losses + rowwise[0]
    assert losses.shape == (n,), f"{losses.shape} =/= {(n,)}"

    argmin: UInt16[Array, ""] = jnp.argmin(losses)
    assert argmin.shape == (), f"{argmin.shape} =/= ()"

    r_loss: Float32[Array, ""] = losses[argmin]
    assert r_loss.shape == (), f"{r_loss.shape} =/= {()}"

    r_indices: UInt16[Array, "n-1"] = recursed.indices[argmin]
    assert r_indices.shape == (n - 1,), f"{r_indices.shape} =/= {(n - 1,)}"

    r_indices: UInt16[Array, "n-1"] = jnp.where(
        r_indices < argmin,
        r_indices,
        r_indices + 1,
    )
    assert r_indices.shape == (n - 1,), f"{r_indices.shape} =/= {(n - 1,)}"

    r_indices: UInt16[Array, "n"] = jnp.concat([argmin[jnp.newaxis], r_indices])
    assert r_indices.shape == (n,), f"{r_indices.shape} =/= {(n,)}"

    r_flip: Bool[Array, "n-1"] = recursed.flip[argmin]
    assert r_flip.shape == (n - 1,), f"{r_flip.shape} =/= {(n - 1,)}"

    r_flip: Bool[Array, "n"] = jnp.concat([flip[0, argmin, jnp.newaxis], r_flip])
    assert r_flip.shape == (n,), f"{r_flip.shape} =/= {(n,)}"

    print(
        f"Compiling {actual.shape}-{ideal.shape}-{rowwise.shape}-{flip.shape} permutation-cruncher..."
    )
    return Permutation(indices=r_indices, flip=r_flip), r_loss


@jit()
def find_permutation(
    actual: Float32[Array, "n m"],
    ideal: Float32[Array, "n m"],
) -> Permutation:
    """
    Exhaustive search for layer-wise permutations minimizing a given loss
    that nonetheless, when all applied in order, reverse any intermediate permutations
    and output the correct indices in their original positions.

    NOTE: INPUT CANNOT BE DIFFERENTIATED.
    TODO: search a bit more for how to detect the above (nothing yet) . . .
    """
    n, _ = actual.shape
    actual_std = jnp.sqrt(
        jnp.sum(jnp.square(stop_gradient(actual)), axis=1, keepdims=True)
    )
    ideal_std = jnp.sqrt(
        jnp.sum(jnp.square(stop_gradient(ideal)), axis=1, keepdims=True)
    )
    actual = actual / (actual_std + 1e-8)
    ideal = ideal / (ideal_std + 1e-8)

    # Create a matrix distancing each row from each other row and its negation:
    stack_neg = lambda x: jnp.stack([x, -x], axis=1)[:, jnp.newaxis]
    rowwise = jnp.abs(
        ideal.astype(jnp.float32)[jnp.newaxis, :, jnp.newaxis]
        - stack_neg(actual.astype(jnp.float32))
    )
    assert rowwise.shape[:3] == (n, n, 2)
    rowwise = jnp.sum(rowwise.reshape(n, n, 2, -1), axis=-1)
    assert rowwise.shape == (n, n, 2)

    flip = jnp.array(jnp.argmin(rowwise, axis=-1), dtype=jnp.bool)
    assert flip.shape == (n, n), f"{flip.shape} =/= {(n, n)}"
    rowwise = jnp.where(flip, rowwise[..., 1], rowwise[..., 0])
    assert rowwise.shape == (n, n), f"{rowwise.shape} =/= {(n, n)}"

    permutation, _ = find_permutation_rec(actual, ideal, rowwise, flip)
    return permutation


@jit()
def layer_distance(
    actual: Weights,
    ideal: Weights,
) -> Tuple[Float32[Array, ""], List[Permutation]]:
    """
    Compute the "true" distance between two sets of weights and biases,
    allowing permutations at every layer without changing the final output.
    Return value: `loss, permutations`
    This function automatically finds the set of permutations minimizing L2 loss,
    but it should be noted that this is only a (very good) approximation:
    the permutations for each layer are computed separately and chained,
    and, technically, there could be mathematical complications related to
    a set of permutations for adjacent layers that would be optimal for both
    yet non-optimal for each layer considered alone.
    In practice, however, the extra loss in situations like the above
    should be entirely negligible.
    TODO: Investigate the above . . . if you have the compute to do so.
    """

    n = layers(actual)
    assert layers(ideal) == n, f"{layers(ideal)} =/= {n}"

    # TODO: run the below (up to the differentiable stuff) through `numba`
    wb_actual = wb(actual)
    wb_ideal = wb(ideal)

    last_p: Optional[Permutation] = None
    permutations = []
    for i in range(n - 1):
        # Why (... - 1) above? b/c we can't change output rows' meaning by permuting them
        # Why loop instead of vectorize? b/c we need to permute columns of the next layer
        ai = stop_gradient(wb_actual[i]).astype(jnp.float32)
        ii = stop_gradient(
            wb_ideal[i]
            if last_p is None
            else jnp.concat(
                [
                    permute(ideal.W[i], last_p, 1),
                    ideal.B[i, ..., jnp.newaxis],
                ],
                axis=-1,
            )
        ).astype(jnp.float32)
        p = find_permutation(ai, ii)
        permutations.append(p)
        last_p = p

    # Calculate loss differentiably:
    ideal = permute_hidden_layers(ideal, permutations)
    wb_a = wb(actual)
    print("wb_a")
    print(wb_a)
    wb_i = wb(ideal)
    print("wb_i")
    print(wb_i)
    std_a = jnp.sqrt(jnp.sum(jnp.square(stop_gradient(wb_a)), axis=-1, keepdims=True))
    print("std_a")
    print(std_a)
    std_i = jnp.sqrt(jnp.sum(jnp.square(stop_gradient(wb_i)), axis=-1, keepdims=True))
    print("std_i")
    print(std_i)
    normalized_a = wb_a / (std_a + 1e-8)
    print("normalized_a")
    print(normalized_a)
    normalized_i = wb_i / (std_i + 1e-8)
    print("normalized_i")
    print(normalized_i)
    L = jnp.sum(jnp.abs(normalized_i - normalized_a))

    return L, permutations
