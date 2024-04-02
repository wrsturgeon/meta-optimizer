from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]
    momentum: Float[Array, ""]
    overstep: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    last_update: Weights
    actual: Weights


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(lr=jnp.array(0.01), momentum=jnp.array(0.9), overstep=jnp.array(0.9))


@jaxtyped(typechecker=beartype)
def init(initial_weights: Weights, p: Params) -> State:
    return State(
        last_update=initial_weights.map(jnp.zeros_like, jnp.zeros_like),
        actual=initial_weights.map(jnp.zeros_like, jnp.zeros_like),
    )


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: Weights,
    dLdw: Weights,
) -> Tuple[State, Weights]:
    assert w.layers() == dLdw.layers() == s.last_update.layers() == s.actual.layers()
    last = dLdw.combine(
        s.last_update,
        lambda di, lu: p.lr * di - p.momentum * lu,
        lambda di, lu: p.lr * di - p.momentum * lu,
    )
    updated = w.combine(last, lambda wi, lu: wi - lu, lambda bi, lu: bi - lu)
    return (
        State(last_update=last, actual=updated),
        updated.combine(
            last,
            lambda wi, lu: wi - p.overstep * lu,
            lambda bi, lu: bi - p.overstep * lu,
        ),
    )
