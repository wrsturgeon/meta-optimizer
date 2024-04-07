from metaoptimizer.optimizers import inverse_sigmoid

from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import nn as jnn, numpy as jnp
from jax.experimental.checkify import check
from jax.tree_util import tree_map, tree_reduce
from jaxtyping import jaxtyped, Array, Float, PyTree
import operator


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    log_lr: Float[Array, ""]
    inv_sig_moving_average_decay: Float[Array, ""]
    inv_sig_moving_square_decay: Float[Array, ""]
    log_epsilon: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    moving_average: PyTree[Float[Array, "..."]]
    correction_average: Float[Array, ""]
    moving_square: PyTree[Float[Array, "..."]]
    correction_square: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def defaults(
    lr: Float[Array, ""] = jnp.array(0.01),
    moving_average_decay: Float[Array, ""] = jnp.array(0.9),
    moving_square_decay: Float[Array, ""] = jnp.array(0.999),
    epsilon: Float[Array, ""] = jnp.array(1e-8),
) -> Params:
    return Params(
        log_lr=jnp.log(lr),
        inv_sig_moving_average_decay=inverse_sigmoid(moving_average_decay),
        inv_sig_moving_square_decay=inverse_sigmoid(moving_square_decay),
        log_epsilon=jnp.log(epsilon),
    )


@jaxtyped(typechecker=beartype)
def init(initial_weights: PyTree[Float[Array, "..."]], p: Params) -> State:
    return State(
        moving_average=tree_map(jnp.zeros_like, initial_weights),
        correction_average=jnn.sigmoid(p.inv_sig_moving_average_decay),
        moving_square=tree_map(jnp.zeros_like, initial_weights),
        correction_square=jnn.sigmoid(p.inv_sig_moving_square_decay),
    )


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Tuple[State, PyTree[Float[Array, "..."]]]:
    lr = jnp.exp(p.log_lr)
    moving_average_decay = jnn.sigmoid(p.inv_sig_moving_average_decay)
    moving_square_decay = jnn.sigmoid(p.inv_sig_moving_square_decay)
    epsilon = jnp.exp(p.log_epsilon)

    persistent_avg = tree_map(lambda w: moving_average_decay * w, s.moving_average)
    novel_avg = tree_map(lambda w: (1.0 - moving_average_decay) * w, dLdw)
    raw_moving_avg = tree_map(operator.add, persistent_avg, novel_avg)
    moving_avg = tree_map(lambda wi: wi / (1.0 - s.correction_average), raw_moving_avg)
    squared = tree_map(jnp.square, dLdw)
    persistent_sq = tree_map(lambda w: moving_square_decay * w, s.moving_square)
    novel_sq = tree_map(lambda w: (1.0 - moving_square_decay) * w, squared)
    raw_moving_sq = tree_map(operator.add, persistent_sq, novel_sq)
    moving_sq = tree_map(lambda wi: wi / (1.0 - s.correction_square), raw_moving_sq)
    rms = tree_map(jnp.sqrt, moving_sq)
    update = tree_map(lambda mi, vi: lr * mi / (vi + epsilon), moving_avg, rms)
    updated = tree_map(operator.sub, w, update)
    return (
        State(
            moving_average=raw_moving_avg,
            correction_average=s.correction_average
            * jnp.clip(
                moving_average_decay,
                a_min=jnp.zeros_like(moving_average_decay),
                a_max=jnp.ones_like(moving_average_decay),
            ),
            moving_square=raw_moving_sq,
            correction_square=s.correction_square
            * jnp.clip(
                moving_square_decay,
                a_min=jnp.zeros_like(moving_square_decay),
                a_max=jnp.ones_like(moving_square_decay),
            ),
        ),
        updated,
    )
