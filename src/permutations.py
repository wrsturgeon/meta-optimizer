from nontest import jit

from beartype import beartype
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float, UInt


@jit
@jaxtyped(typechecker=beartype)
def permute(x: Float[Array, "n ..."], indices: UInt[Array, "n"]) -> Array:
    # TODO: disable these assertions in production
    n = x.shape[0]
    assert indices.shape == (n,)
    assert jnp.all(indices < n)
    for i in range(n):
        assert not jnp.isin(indices[i], indices[:i])
    return x[indices]


@jit
@jaxtyped(typechecker=beartype)
def score(
    Wactual: list[Float[Array, "..."]],
    Bactual: list[Float[Array, "..."]],
    Wideal: list[Float[Array, "..."]],
    Bideal: list[Float[Array, "..."]],
    permutations: list[list[int]],
) -> jnp.float32:
    DOF = len(permutations)
    # You can think of layers in a chain held together with twistable string;
    # I mean DOF (degrees of freedom) in the sense that
    # we can twist (rotate) each inter-layer joint however we want,
    # but the last joint has to be twisted all the way back to the original position.
    # Hence `+ 1` below:
    assert DOF + 1 == len(Wactual) == len(Bactual) == len(Wideal) == len(Bideal)
    # The last rotation has to undo all the others for output indices to make sense:
    # cumulative_undo = ...
    raise NotImplementedError()


@jit
@jaxtyped(typechecker=beartype)
def find_layer_wise_permutations(
    Wactual: list[Float[Array, "..."]],
    Bactual: list[Float[Array, "..."]],
    Wideal: list[Float[Array, "..."]],
    Bideal: list[Float[Array, "..."]],
    last_best: list[list[int]],
) -> tuple[list[list[int]], jnp.float32]:
    """
    Exhaustive search for layer-wise permutations minimizing a given loss
    that nonetheless, when all applied in order, reverse any intermediate permutations
    and output the correct indices in their original positions.
    This not quite as bad as naïve brute-force search (unfortunately close), but
    we maintain a running best-yet score and short-circuit anything above it.
    Better yet, since these matrices should change slowly,
    we initialize the best-yet score with the
    best matrix from the last step.
    """
    # See comment in `score` for an explanation of `DOF` below
    DOF = len(last_best)
    running_min = score(Wactual, Bactual, Wideal, Bideal, last_best)
    # TODO: Breadth-first search away from `last_best` instead of simple iteration
    raise NotImplementedError()
