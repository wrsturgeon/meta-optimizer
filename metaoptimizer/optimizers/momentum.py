from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import jaxtyped, Array, Float, PyTree


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]
    momentum: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    last_update: PyTree[Float[Array, "..."]]


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(lr=jnp.array(0.01), momentum=jnp.array(0.9))


@jaxtyped(typechecker=beartype)
def init(initial_weights: PyTree[Float[Array, "..."]], p: Params) -> State:
    return State(last_update=tree_map(jnp.zeros_like, initial_weights))


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Tuple[State, PyTree[Float[Array, "..."]]]:
    last = tree_map(lambda di, lu: p.lr * di - p.momentum * lu, dLdw, s.last_update)
    updated = tree_map(lambda wi, lu: wi - lu, w, last)
    return State(last_update=last), updated
