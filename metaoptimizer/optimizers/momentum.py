from metaoptimizer.distributions import inverse_sigmoid

from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import nn as jnn, numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import jaxtyped, Array, Float, PyTree
import operator


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    log_lr: Float[Array, ""]
    inv_sig_momentum: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    last_update: PyTree[Float[Array, "..."]]


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(
        log_lr=jnp.log(0.01), inv_sig_momentum=inverse_sigmoid(jnp.array(0.9))
    )


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
    lr = jnp.exp(p.log_lr)
    momentum = jnn.sigmoid(p.inv_sig_momentum)
    last = tree_map(lambda di, lu: lr * di - momentum * lu, dLdw, s.last_update)
    updated = tree_map(operator.sub, w, last)
    return State(last_update=last), updated
