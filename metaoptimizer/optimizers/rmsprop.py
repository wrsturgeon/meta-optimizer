from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float


@jaxtyped(typechecker=beartype)
class Params(NamedTuple):
    lr: Float[Array, ""]
    moving_square_decay: Float[Array, ""]


@jaxtyped(typechecker=beartype)
class State(NamedTuple):
    moving_square: Weights


@jaxtyped(typechecker=beartype)
def defaults() -> Params:
    return Params(lr=jnp.array(0.01), moving_square_decay=jnp.array(0.9))


@jaxtyped(typechecker=beartype)
def init(initial_weights: Weights, p: Params) -> State:
    return State(moving_square=initial_weights.map(jnp.zeros_like, jnp.zeros_like))


@jaxtyped(typechecker=beartype)
def update(
    p: Params,
    s: State,
    w: Weights,
    dLdw: Weights,
) -> Tuple[State, Weights]:
    assert w.layers() == dLdw.layers() == s.moving_square.layers()
    squared = dLdw.map(jnp.square, jnp.square)
    persistent_sq = s.moving_square.map(
        lambda w: p.moving_square_decay * w,
        lambda b: p.moving_square_decay * b,
    )
    novel_sq = squared.map(
        lambda w: (1.0 - p.moving_square_decay) * w,
        lambda b: (1.0 - p.moving_square_decay) * b,
    )
    moving_sq = persistent_sq.combine(
        novel_sq,
        lambda wi, ni: wi + ni,
        lambda bi, ni: bi + ni,
    )
    rms = moving_sq.map(jnp.sqrt, jnp.sqrt)
    update = dLdw.combine(
        rms,
        lambda di, ri: p.lr * di / ri,
        lambda di, ri: p.lr * di / ri,
    )
    updated = w.combine(update, lambda wi, ui: wi - ui, lambda bi, ui: bi - ui)
    return State(moving_square=moving_sq), updated
