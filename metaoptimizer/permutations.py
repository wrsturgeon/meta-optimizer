from metaoptimizer.nontest import jit
from metaoptimizer.weights import Weights

from beartype import beartype
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Bool, Float, UInt
from typing import NamedTuple


@jit
@jaxtyped(typechecker=beartype)
def permute(x: Float[Array, "..."], indices: UInt[Array, "n"], axis: int) -> Array:
    # TODO: disable these assertions in production
    n = x.shape[axis]
    assert indices.shape == (n,)
    # assert jnp.all(indices >= 0) # implicit in the type signature
    assert jnp.all(indices < n)
    for i in range(n):
        assert not jnp.isin(indices[i], indices[:i])
    return jnp.apply_along_axis(lambda z: z[indices], axis, x)


@jaxtyped(typechecker=beartype)
class Permutation(NamedTuple):
    """Description of a permutation on some tensor (without explicitly carrying around that tensor)."""

    indices: UInt[Array, "n"]
    flip: Bool[Array, "n"]
    loss: Float[Array, ""]


@jit
@jaxtyped(typechecker=beartype)
def score_permutation(
    Wactual: Float[Array, "n_out n_in"],
    Bactual: Float[Array, "n_out"],
    Wideal: Float[Array, "n_out n_in"],
    Bideal: Float[Array, "n_out"],
    indices: UInt[Array, "n_out"],
) -> Permutation:
    n_out, n_in = Wactual.shape
    assert all([0 <= i < n_out for i in indices])
    assert all([i not in indices[:i] for i in indices])
    assert jnp.allclose(1, jnp.mean(jnp.square(Wactual), axis=1))
    assert jnp.allclose(1, jnp.mean(jnp.square(Wideal), axis=1))
    Wperm = permute(Wactual, indices, 0)
    Bperm = permute(Bactual, indices, 0)
    loss_orig = jnp.sum(jnp.square(Wideal - Wperm), axis=1) + jnp.square(Bideal - Bperm)
    loss_flip = jnp.sum(jnp.square(Wideal + Wperm), axis=1) + jnp.square(Bideal + Bperm)
    assert loss_orig.shape == loss_flip.shape == (n_out,)
    flip = jnp.array(jnp.argmin(jnp.array([loss_orig, loss_flip]), axis=0), dtype=bool)
    assert flip.shape == (n_out,)
    return Permutation(
        indices=indices,
        flip=flip,
        loss=jnp.sum(jnp.where(flip, loss_orig, loss_flip)),
    )


@jit
@jaxtyped(typechecker=beartype)
def find_permutation_rec(
    Wactual: Float[Array, "n_out n_in"],
    Bactual: Float[Array, "n_out"],
    Wideal: Float[Array, "n_out n_in"],
    Bideal: Float[Array, "n_out"],
    best: Permutation,
    acc: Permutation,
) -> Permutation:
    n_out, n_in = Wactual.shape
    assert best.indices.shape == best.flip.shape == (n_out,)
    (acc_len,) = acc.indices.shape
    assert acc.flip.shape == (acc_len,)
    if acc_len >= n_out:
        assert acc_len == n_out
        return acc
    # implicit else
    w_actual = Wactual[acc_len]
    b_actual = Bactual[acc_len]
    # TODO: Breadth-first search away from `best` instead of simple iteration
    for i in range(n_out):
        if i not in acc.indices:
            w_ideal = Wideal[i]
            b_ideal = Bideal[i]
            indices = jnp.append(acc.indices, jnp.array(i, dtype=jnp.uint32))
            loss_orig = jnp.sum(jnp.square(w_ideal - w_actual)) + jnp.square(
                b_ideal - b_actual
            )
            loss_flip = jnp.sum(jnp.square(w_ideal + w_actual)) + jnp.square(
                b_ideal + b_actual
            )
            flip = jnp.append(
                acc.flip, jnp.array(loss_flip < loss_orig, dtype=jnp.bool)
            )
            loss = acc.loss + jnp.min(jnp.array([loss_orig, loss_flip]))
            if loss < best.loss:
                current = Permutation(indices=indices, flip=flip, loss=loss)
                best = find_permutation_rec(
                    Wactual, Bactual, Wideal, Bideal, best, current
                )
    return best


@jit
@jaxtyped(typechecker=beartype)
def find_permutation(
    Wactual: Float[Array, "n_out n_in"],
    Bactual: Float[Array, "n_out"],
    Wideal: Float[Array, "n_out n_in"],
    Bideal: Float[Array, "n_out"],
    last_best: UInt[Array, "n_out"],
) -> Permutation:
    """
    Exhaustive search for layer-wise permutations minimizing a given loss
    that nonetheless, when all applied in order, reverse any intermediate permutations
    and output the correct indices in their original positions.
    This not quite as bad as naïve brute-force search (unfortunately close), since
    we maintain a running best-yet loss and short-circuit anything above it.
    Better yet, since these matrices should change slowly,
    we initialize the best-yet loss with the
    best matrix from the last step.
    """
    n_out, n_in = Wactual.shape
    Sactual = jnp.sqrt(jnp.mean(jnp.square(Wactual), axis=1, keepdims=True))
    assert Sactual.shape == (n_out, 1)
    Wactual = Wactual.at[...].divide(Sactual)
    Bactual = Bactual.at[...].divide(Sactual[:, 0])
    Sideal = jnp.sqrt(jnp.mean(jnp.square(Wideal), axis=1, keepdims=True))
    assert Sideal.shape == (n_out, 1)
    Wideal = Wideal.at[...].divide(Sideal)
    Bideal = Bideal.at[...].divide(Sideal[:, 0])
    return find_permutation_rec(
        Wactual,
        Bactual,
        Wideal,
        Bideal,
        score_permutation(Wactual, Bactual, Wideal, Bactual, last_best),
        Permutation(
            jnp.array([], dtype=jnp.uint32),
            jnp.array([], dtype=jnp.bool),
            jnp.array(0, dtype=jnp.float32),
        ),
    )


@jit
@jaxtyped(typechecker=beartype)
def permute_hidden_layers(
    w: Weights,
    ps: list[UInt[Array, "..."]],
) -> Weights:
    """Permute hidden layers' columns locally without changing the output of a network."""
    n = len(ps)
    assert w.layers() == n + 1
    for i in range(n):
        p = ps[i]
        w.W[i] = permute(w.W[i], p, axis=0)
        w.W[i + 1] = permute(w.W[i + 1], p, axis=1)
        w.B[i] = permute(w.B[i], p, axis=0)
    return w


@jit
@jaxtyped(typechecker=beartype)
def layer_distance(
    actual: Weights,
    ideal: Weights,
    last_best: list[UInt[Array, "..."]],
) -> tuple[Float[Array, ""], list[Permutation]]:
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
    assert ideal.layers() == len(last_best) + 1 == n
    ps = [
        find_permutation(wa, ba, wi, bi, lb)
        for wa, ba, wi, bi, lb in zip(actual.W, actual.B, ideal.W, ideal.B, last_best)
    ]
    return sum([p.loss for p in ps]), ps
