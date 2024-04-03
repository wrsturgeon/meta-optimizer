from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_structure
from jaxtyping import jaxtyped, Array, Float, PyTree


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    pass


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(lr=jnp.array(0.01))


@jaxtyped(typechecker=beartype)
def init(initial_weights: PyTree[Float[Array, "..."]], p: Params) -> State:
    return State()


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Tuple[State, PyTree[Float[Array, "..."]]]:
    updated = tree_map(lambda wi, di: wi - p.lr * di, w, dLdw)
    return State(), updated
