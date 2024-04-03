from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import jaxtyped, Array, Float, PyTree


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]
    moving_square_decay: Float[Array, ""]
    epsilon: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    moving_square: PyTree[Float[Array, "..."]]


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(
        lr=jnp.array(0.01),
        moving_square_decay=jnp.array(0.9),
        epsilon=jnp.array(1e-8),
    )


@jaxtyped(typechecker=beartype)
def init(initial_weights: PyTree[Float[Array, "..."]], p: Params) -> State:
    return State(moving_square=tree_map(jnp.zeros_like, initial_weights))


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Tuple[State, PyTree[Float[Array, "..."]]]:
    squared = tree_map(jnp.square, dLdw)
    persistent_sq = tree_map(lambda w: p.moving_square_decay * w, s.moving_square)
    novel_sq = tree_map(lambda w: (1.0 - p.moving_square_decay) * w, squared)
    moving_sq = tree_map(lambda a, b: a + b, persistent_sq, novel_sq)
    rms = tree_map(jnp.sqrt, moving_sq)
    update = tree_map(lambda di, ri: p.lr * di / (ri + p.epsilon), dLdw, rms)
    updated = tree_map(lambda wi, ui: wi - ui, w, update)
    return State(moving_square=moving_sq), updated
