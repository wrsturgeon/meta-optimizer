from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import List, Tuple
from jax import nn as jnn, numpy as jnp, pmap
from jax.experimental.checkify import check
from jax.lax import cond
from jaxtyping import jaxtyped, Array, Bool, Float, PyTree, UInt
from typing import NamedTuple


@jaxtyped(typechecker=beartype)
def permute(x: Float[Array, "..."], indices: UInt[Array, "n"], axis: int) -> Array:
    # TODO: disable these assertions in production
    n = x.shape[axis]
    assert indices.shape == (n,)  # needs to be checked only once
    check(
        jnp.all(indices >= 0),
        "{indices} must be nonnegative everywhere",
        indices=indices,
    )  # implicit in the type signature
    check(
        jnp.all(indices < n),
        "{indices} must be less than {n} everywhere",
        indices=indices,
        n=jnp.array(n),
    )
    for i in range(indices.size):
        check(
            jnp.logical_not(jnp.isin(indices[i], indices[:i])),
            "Each element in {indices} must be unique (#{i} is not)",
            indices=indices,
            i=jnp.array(i),
        )
    return jnp.apply_along_axis(lambda z: z[indices], axis, x)


@jaxtyped(typechecker=beartype)
class Permutation(NamedTuple):
    """Description of a permutation on some tensor (without explicitly carrying around that tensor)."""

    indices: UInt[Array, "n"]
    flip: Bool[Array, "n"]
    loss: Float[Array, ""]
    # TODO: REPLACE `flip` WITH GENERALIZED +/- COEFFICIENTS (VERY NECESSARY)


@jaxtyped(typechecker=beartype)
def find_permutation_rec(
    actual: Float[Array, "n ..."],
    ideal: Float[Array, "n ..."],
    rowwise: Float[Array, "n n"],
    flip: Bool[Array, "n n"],
) -> Permutation:
    # Note that, in the last two matrices above,
    # the 1st axis represents `_actual`, and
    # the 2nd axis represents `_ideal`, so
    # `rowwise[i, j, k]` is the distance between
    # `actual[i]` and `ideal[j]` (w/ the former flipped iff `k`).
    # Note, further, that we're permuting `actual`,
    # so we should really only ever care about `actual[n]`
    # where `n` is the recursive depth thus far.
    # In other words, we can just delete the top row
    # in each recursive call and not keep track of `n`.

    n = actual.shape[0]

    if n == 0:
        return Permutation(
            indices=jnp.array([], dtype=jnp.uint32),
            flip=jnp.array([], dtype=jnp.bool),
            loss=jnp.array(0.0, dtype=jnp.float32),
        )

    # Basic idea is to select the first index of the final permutation,
    # remove the element at that index from the array we'll work with,
    # recurse, increment all indices greater than the one we chose,
    # then `cons` it onto the first index we just chose.
    @jaxtyped(typechecker=beartype)
    def recurse_on_index(i: int) -> Permutation:
        def without_i(x):
            return jnp.concat([x[:i], x[(i + 1) :]])

        def without2d(x):
            # See the comment at the top of this function body
            # for an explanation of `[1:, ...]`
            return jnp.concat([x[1:, :i], x[1:, (i + 1) :]], axis=1)

        recurse = find_permutation_rec(
            without_i(actual),
            without_i(ideal),
            without2d(rowwise),
            without2d(flip),
        )
        indices = recurse.indices
        indices = jnp.where(indices < i, indices, indices + 1)
        indices = jnp.concat([jnp.array([i], dtype=jnp.uint32), indices])
        return Permutation(
            indices=indices,
            flip=jnp.concat([flip[0, i, None], recurse.flip]),
            loss=recurse.loss + rowwise[0, i],
        )

    # TODO: Would `vmap`/`pmap` work better? (it would have to store everything...)
    best = recurse_on_index(0)
    for i in range(1, n):
        new = recurse_on_index(i)
        best = cond(new.loss < best.loss, lambda: new, lambda: best)
    return best


@jaxtyped(typechecker=beartype)
def find_permutation(
    actual: Float[Array, "n ..."],
    ideal: Float[Array, "n ..."],
) -> Permutation:
    """
    Exhaustive search for layer-wise permutations minimizing a given loss
    that nonetheless, when all applied in order, reverse any intermediate permutations
    and output the correct indices in their original positions.
    """
    n = actual.shape[0]
    actual = actual.reshape(n, -1)
    ideal = ideal.reshape(n, -1)
    actual = actual / jnp.sqrt(jnp.mean(jnp.square(actual), axis=1, keepdims=True))
    ideal = ideal / jnp.sqrt(jnp.mean(jnp.square(ideal), axis=1, keepdims=True))

    # Create a matrix distancing each row from each other row and its negation:
    stack_neg = lambda x: jnp.stack([x, -x], axis=1)[:, jnp.newaxis]
    rowwise = jnp.abs(ideal[jnp.newaxis, :, jnp.newaxis] - stack_neg(actual))
    assert rowwise.shape[:3] == (n, n, 2)
    rowwise = jnp.sum(rowwise.reshape(n, n, 2, -1), axis=-1)
    assert rowwise.shape == (n, n, 2)

    flip = jnp.array(jnp.argmin(rowwise, axis=-1), dtype=jnp.bool)
    assert flip.shape == (n, n), f"{flip.shape} =/= {(n, n)}"
    rowwise = jnp.where(flip, rowwise[..., 1], rowwise[..., 0])
    assert rowwise.shape == (n, n), f"{rowwise.shape} =/= {(n, n)}"

    return find_permutation_rec(actual, ideal, rowwise, flip)


@jaxtyped(typechecker=beartype)
def find_permutation_weights(
    Wactual: Float[Array, "n_out n_in"],
    Bactual: Float[Array, "n_out"],
    Wideal: Float[Array, "n_out n_in"],
    Bideal: Float[Array, "n_out"],
) -> Permutation:
    return find_permutation(
        jnp.concat([Wactual, Bactual[..., jnp.newaxis]], axis=-1),
        jnp.concat([Wideal, Bideal[..., jnp.newaxis]], axis=-1),
    )


@jaxtyped(typechecker=beartype)
def permute_hidden_layers(
    w: Weights,
    ps: List[UInt[Array, "..."]],
) -> PyTree[Float[Array, "..."]]:
    """Permute hidden layers' columns locally without changing the output of a network."""
    n = len(ps)
    assert w.layers() == n + 1
    for i in range(n):
        p = ps[i]
        w.W[i] = permute(w.W[i], p, axis=0)
        w.W[i + 1] = permute(w.W[i + 1], p, axis=1)
        w.B[i] = permute(w.B[i], p, axis=0)
    return w


@jaxtyped(typechecker=beartype)
def layer_distance(
    actual: Weights,
    ideal: Weights,
) -> Tuple[Float[Array, ""], List[Permutation]]:
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
    n = actual.layers()
    assert ideal.layers() == n, f"{ideal.layers()} =/= {n}"
    ps = [
        find_permutation_weights(wa, ba, wi, bi)
        for wa, ba, wi, bi in zip(
            actual.W[:-1],
            actual.B[:-1],
            ideal.W[:-1],
            ideal.B[:-1],
            # Why [:-1]? Cuz we can't change output rows' meaning by permuting them
        )
    ]
    return sum([p.loss for p in ps]), ps
