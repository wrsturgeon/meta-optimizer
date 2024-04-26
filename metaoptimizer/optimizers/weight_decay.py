from metaoptimizer.optimizers import inverse_sigmoid

from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import jaxtyped, Array, Float, PyTree


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    log_lr: Float[Array, ""]
    inv_sig_weight_decay: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    pass


@jaxtyped(typechecker=beartype)
def defaults(
    lr: Float[Array, ""] = jnp.array(0.01),
    weight_decay: Float[Array, ""] = jnp.array(0.999),
) -> Params:
    return Params(
        log_lr=jnp.log(lr),
        inv_sig_weight_decay=inverse_sigmoid(weight_decay),
    )


@jaxtyped(typechecker=beartype)
def init(initial_weights: PyTree[Float[Array, "..."]], p: Params) -> State:
    return State()


# @check_and_compile()
@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Tuple[State, PyTree[Float[Array, "..."]]]:
    lr = jnp.exp(p.log_lr)
    weight_decay = jnn.sigmoid(p.inv_sig_weight_decay)
    # TODO: Find a generalizable way to apply weight decay only to weights, not to biases
    updated = tree_map(lambda wi, di: weight_decay * wi - lr * di, w, dLdw)
    return State(), updated
