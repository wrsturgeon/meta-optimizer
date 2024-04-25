from metaoptimizer.optimizers import inverse_sigmoid

from beartype import beartype
from beartype.typing import NamedTuple, Tuple, Union
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp
from jax.experimental.checkify import check
from jax.tree_util import tree_map, tree_reduce
from jaxtyping import jaxtyped, Array, Float, Float64, PyTree
import operator


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    log_lr: Float64[Array, ""]
    inv_sig_moving_average_decay: Float64[Array, ""]
    inv_sig_moving_square_decay: Float64[Array, ""]
    inv_sig_moving_square_quotient: Float64[Array, ""]
    inv_sig_momentum: Float64[Array, ""]
    log_overstep: Float64[Array, ""]
    inv_sig_weight_decay: Float64[Array, ""]
    log_epsilon: Float64[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    last_update: PyTree[Float[Array, "..."]]
    actual: PyTree[Float[Array, "..."]]
    moving_average: PyTree[Float[Array, "..."]]
    correction_average: Float[Array, ""]
    moving_square: PyTree[Float[Array, "..."]]
    correction_square: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def defaults(
    lr: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
    moving_average_decay: Float64[Array, ""] = jnp.array(0.9, dtype=jnp.float64),
    moving_square_decay: Float64[Array, ""] = jnp.array(0.999, dtype=jnp.float64),
    moving_square_quotient: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
    momentum: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
    overstep: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
    weight_decay: Float64[Array, ""] = jnp.array(1 - 1e-8, dtype=jnp.float64),
    epsilon: Float64[Array, ""] = jnp.array(1e-8, dtype=jnp.float64),
) -> Params:
    return Params(
        log_lr=jnp.log(lr),
        inv_sig_moving_average_decay=inverse_sigmoid(moving_average_decay),
        inv_sig_moving_square_decay=inverse_sigmoid(moving_square_decay),
        inv_sig_moving_square_quotient=inverse_sigmoid(moving_square_quotient),
        inv_sig_momentum=inverse_sigmoid(momentum),
        log_overstep=jnp.log(overstep),
        inv_sig_weight_decay=inverse_sigmoid(weight_decay),
        log_epsilon=jnp.log(epsilon),
    )


@jaxtyped(typechecker=beartype)
def init(initial_weights: PyTree[Float[Array, "..."]], p: Params) -> State:
    return State(
        last_update=tree_map(jnp.zeros_like, initial_weights),
        actual=tree_map(jnp.zeros_like, initial_weights),
        moving_average=tree_map(jnp.zeros_like, initial_weights),
        correction_average=jnn.sigmoid(p.inv_sig_moving_average_decay),
        moving_square=tree_map(jnp.zeros_like, initial_weights),
        correction_square=jnn.sigmoid(p.inv_sig_moving_square_decay),
    )


@jaxtyped(typechecker=beartype)
def flatten_quotient(
    x: Float[Array, "*n"],
    k: Union[Float[Array, ""], Float[Array, "*n"]],
) -> Float[Array, "*n"]:
    # Idea is to take a scalar from 0 to 1 and flatten a signal that will act as a quotient.
    # Since we're dividing with it, we want "no influence" to mean "1 everywhere," not "0 everywhere."
    return 1 + (k * (x - 1))


# @check_and_compile()
def update(
    p: Params,
    s: State,
    w: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Tuple[State, PyTree[Float[Array, "..."]]]:
    lr = jnp.exp(p.log_lr)
    moving_average_decay = jnn.sigmoid(p.inv_sig_moving_average_decay)
    moving_square_decay = jnn.sigmoid(p.inv_sig_moving_square_decay)
    moving_square_quotient = jnn.sigmoid(p.inv_sig_moving_square_quotient)
    momentum = jnn.sigmoid(p.inv_sig_momentum)
    overstep = jnp.exp(p.log_overstep)
    weight_decay = jnn.sigmoid(p.inv_sig_weight_decay)
    epsilon = jnp.exp(p.log_epsilon)

    raw_moving_avg = tree_map(
        lambda moving, current: (
            moving_average_decay * moving + (1 - moving_average_decay) * current
        ),
        s.moving_average,
        dLdw,
    )
    moving_avg = tree_map(lambda x: x / (1 - s.correction_average), raw_moving_avg)
    raw_moving_sq = tree_map(
        lambda moving, current: (
            moving_square_decay * moving
            + (1 - moving_square_decay) * jnp.square(current)
        ),
        s.moving_square,
        dLdw,
    )
    moving_sq = tree_map(lambda x: x / (1 - s.correction_square), raw_moving_sq)
    update = tree_map(
        lambda m_avg, m_sq, last: (
            (
                (lr * m_avg)
                / flatten_quotient(jnp.sqrt(m_sq + epsilon), moving_square_quotient)
            )
            + momentum * last
        ),
        moving_avg,
        moving_sq,
        s.last_update,
    )
    # TODO: Find a generalizable way to apply weight decay only to weights, not to biases
    updated = tree_map(lambda wi, ui: weight_decay * wi - ui, w, update)
    return (
        State(
            last_update=update,
            actual=updated,
            moving_average=raw_moving_avg,
            correction_average=s.correction_average * moving_average_decay,
            moving_square=raw_moving_sq,
            correction_square=s.correction_square * moving_square_decay,
        ),
        tree_map(lambda wi, ui: wi - overstep * ui, updated, update),
    )
