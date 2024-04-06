from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_structure
from jaxtyping import jaxtyped, Array, Float, PyTree


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    log_lr: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    pass


@jaxtyped(typechecker=beartype)
def defaults(lr: Float[Array, ""] = jnp.array(0.01)) -> Params:
    return Params(log_lr=jnp.log(lr))


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
    lr = jnp.exp(p.log_lr)
    updated = tree_map(lambda wi, di: wi - lr * di, w, dLdw)
    return State(), updated
