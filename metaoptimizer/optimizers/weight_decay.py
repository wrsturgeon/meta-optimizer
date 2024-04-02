from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]
    weight_decay: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    pass


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(lr=jnp.array(0.01), weight_decay=jnp.array(0.999))


@jaxtyped(typechecker=beartype)
def init(initial_weights: Weights, p: Params) -> State:
    return State()


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: Weights,
    dLdw: Weights,
) -> Tuple[State, Weights]:
    assert w.layers() == dLdw.layers()
    updated = w.combine(
        dLdw,
        lambda wi, di: p.weight_decay * wi - p.lr * di,
        lambda bi, di: bi - p.lr * di,
    )
    return State(), updated
