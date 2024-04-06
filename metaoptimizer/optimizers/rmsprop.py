from metaoptimizer.distributions import inverse_sigmoid

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
    inv_sig_moving_square_decay: Float[Array, ""]
    log_epsilon: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    moving_square: PyTree[Float[Array, "..."]]


@jaxtyped(typechecker=beartype)
def defaults(
    lr: Float[Array, ""] = jnp.array(0.01),
    moving_square_decay: Float[Array, ""] = jnp.array(0.9),
    epsilon: Float[Array, ""] = jnp.array(1e-8),
) -> Params:
    return Params(
        log_lr=jnp.log(lr),
        inv_sig_moving_square_decay=inverse_sigmoid(moving_square_decay),
        log_epsilon=jnp.log(epsilon),
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
    lr = jnp.exp(p.log_lr)
    moving_square_decay = jnn.sigmoid(p.inv_sig_moving_square_decay)
    epsilon = jnp.exp(p.log_epsilon)
    squared = tree_map(jnp.square, dLdw)
    persistent_sq = tree_map(lambda w: moving_square_decay * w, s.moving_square)
    novel_sq = tree_map(lambda w: (1.0 - moving_square_decay) * w, squared)
    moving_sq = tree_map(operator.add, persistent_sq, novel_sq)
    rms = tree_map(jnp.sqrt, moving_sq)
    update = tree_map(lambda di, ri: lr * di / (ri + epsilon), dLdw, rms)
    updated = tree_map(operator.sub, w, update)
    return State(moving_square=moving_sq), updated
