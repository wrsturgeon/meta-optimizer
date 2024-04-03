from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import jaxtyped, Array, Float, PyTree


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]
    moving_average_decay: Float[Array, ""]
    moving_square_decay: Float[Array, ""]
    epsilon: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    moving_average: PyTree[Float[Array, "..."]]
    correction_average: Float[Array, ""]
    moving_square: PyTree[Float[Array, "..."]]
    correction_square: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(
        lr=jnp.array(0.01),
        moving_average_decay=jnp.array(0.9),
        moving_square_decay=jnp.array(0.999),
        epsilon=jnp.array(1e-8),
    )


@jaxtyped(typechecker=beartype)
def init(initial_weights: PyTree[Float[Array, "..."]], p: Params) -> State:
    return State(
        moving_average=tree_map(jnp.zeros_like, initial_weights),
        correction_average=p.moving_average_decay,
        moving_square=tree_map(jnp.zeros_like, initial_weights),
        correction_square=p.moving_square_decay,
    )


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Tuple[State, PyTree[Float[Array, "..."]]]:
    persistent_avg = tree_map(lambda w: p.moving_average_decay * w, s.moving_average)
    novel_avg = tree_map(lambda w: (1.0 - p.moving_average_decay) * w, dLdw)
    raw_moving_avg = tree_map(lambda a, b: a + b, persistent_avg, novel_avg)
    moving_avg = tree_map(lambda wi: wi / (1.0 - s.correction_average), raw_moving_avg)
    squared = tree_map(jnp.square, dLdw)
    persistent_sq = tree_map(lambda w: p.moving_square_decay * w, s.moving_square)
    novel_sq = tree_map(lambda w: (1.0 - p.moving_square_decay) * w, squared)
    raw_moving_sq = tree_map(lambda a, b: a + b, persistent_sq, novel_sq)
    moving_sq = tree_map(lambda wi: wi / (1.0 - s.correction_square), raw_moving_sq)
    rms = tree_map(jnp.sqrt, moving_sq)
    update = tree_map(lambda mi, vi: p.lr * mi / (vi + p.epsilon), moving_avg, rms)
    updated = tree_map(lambda wi, ui: wi - ui, w, update)
    return (
        State(
            moving_average=raw_moving_avg,
            correction_average=s.correction_average
            * jnp.clip(
                p.moving_average_decay,
                a_min=jnp.zeros_like(p.moving_average_decay),
                a_max=jnp.ones_like(p.moving_average_decay),
            ),
            moving_square=raw_moving_sq,
            correction_square=s.correction_square
            * jnp.clip(
                p.moving_square_decay,
                a_min=jnp.zeros_like(p.moving_square_decay),
                a_max=jnp.ones_like(p.moving_square_decay),
            ),
        ),
        updated,
    )
