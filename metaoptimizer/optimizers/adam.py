from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]
    moving_average_decay: Float[Array, ""]
    moving_square_decay: Float[Array, ""]
    epsilon: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    moving_average: Weights
    correction_average: Float[Array, ""]
    moving_square: Weights
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
def init(initial_weights: Weights, p: Params) -> State:
    return State(
        moving_average=initial_weights.map(jnp.zeros_like, jnp.zeros_like),
        correction_average=p.moving_average_decay,
        moving_square=initial_weights.map(jnp.zeros_like, jnp.zeros_like),
        correction_square=p.moving_square_decay,
    )


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: Weights,
    dLdw: Weights,
) -> Tuple[State, Weights]:
    assert (
        w.layers()
        == dLdw.layers()
        == s.moving_average.layers()
        == s.moving_square.layers()
    )
    persistent_avg = s.moving_average.map(
        lambda w: p.moving_average_decay * w,
        lambda b: p.moving_average_decay * b,
    )
    novel_avg = dLdw.map(
        lambda w: (1.0 - p.moving_average_decay) * w,
        lambda b: (1.0 - p.moving_average_decay) * b,
    )
    raw_moving_avg = persistent_avg.combine(
        novel_avg,
        lambda wi, ni: wi + ni,
        lambda bi, ni: bi + ni,
    )
    moving_avg = raw_moving_avg.map(
        lambda wi: wi / (1.0 - s.correction_average),
        lambda wi: wi / (1.0 - s.correction_average),
    )
    squared = dLdw.map(jnp.square, jnp.square)
    persistent_sq = s.moving_square.map(
        lambda w: p.moving_square_decay * w,
        lambda b: p.moving_square_decay * b,
    )
    novel_sq = squared.map(
        lambda w: (1.0 - p.moving_square_decay) * w,
        lambda b: (1.0 - p.moving_square_decay) * b,
    )
    raw_moving_sq = persistent_sq.combine(
        novel_sq,
        lambda wi, ni: wi + ni,
        lambda bi, ni: bi + ni,
    )
    moving_sq = raw_moving_sq.map(
        lambda wi: wi / (1.0 - s.correction_square),
        lambda wi: wi / (1.0 - s.correction_square),
    )
    rms = moving_sq.map(jnp.sqrt, jnp.sqrt)
    update = moving_avg.combine(
        rms,
        lambda mi, vi: p.lr * mi / (vi + p.epsilon),
        lambda mi, vi: p.lr * mi / (vi + p.epsilon),
    )
    updated = w.combine(update, lambda wi, ui: wi - ui, lambda bi, ui: bi - ui)
    return (
        State(
            moving_average=raw_moving_avg,
            correction_average=s.correction_average * p.moving_average_decay,
            moving_square=raw_moving_sq,
            correction_square=s.correction_square * p.moving_square_decay,
        ),
        updated,
    )
