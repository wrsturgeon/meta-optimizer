from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Self
from jax import jit, numpy as jnp
from jaxtyping import jaxtyped, Array, Float
from typing import NamedTuple


@jaxtyped(typechecker=beartype)
class SGD(NamedTuple):
    lr: Float[Array, ""]

    @jit
    def __call__(self, w: Weights, dLdw: Weights) -> tuple[Self, Weights]:
        assert w.layers() == dLdw.layers()
        updated = w.combine(
            dLdw,
            lambda wi, di: wi - self.lr * di,
            lambda bi, di: bi - self.lr * di,
        )
        return self, updated


@jaxtyped(typechecker=beartype)
class WeightDecay(NamedTuple):
    lr: Float[Array, ""]

    weight_decay: Float[Array, ""]

    @jit
    def __call__(self, w: Weights, dLdw: Weights) -> tuple[Self, Weights]:
        assert w.layers() == dLdw.layers()
        updated = w.combine(
            dLdw,
            lambda wi, di: self.weight_decay * wi - self.lr * di,
            lambda bi, di: bi - self.lr * di,
        )
        return self, updated


@jaxtyped(typechecker=beartype)
class Momentum(NamedTuple):
    lr: Float[Array, ""]

    momentum: Float[Array, ""]
    last_update: Weights

    @jit
    def __call__(self, w: Weights, dLdw: Weights) -> tuple[Self, Weights]:
        assert (
            w.layers() == dLdw.layers() == self.last_update.layers()
        ), f"{w.layers()} == {dLdw.layers()} == {self.last_update.layers()}"
        last = dLdw.combine(
            self.last_update,
            lambda di, lu: self.lr * di - self.momentum * lu,
            lambda di, lu: self.lr * di - self.momentum * lu,
        )
        updated = w.combine(last, lambda wi, lu: wi - lu, lambda bi, lu: bi - lu)
        return self._replace(last_update=last), updated


@jaxtyped(typechecker=beartype)
class Nesterov(NamedTuple):
    lr: Float[Array, ""]

    momentum: Float[Array, ""]
    last_update: Weights

    overstep: Float[Array, ""]
    actual: Weights

    @jit
    def __call__(self, w: Weights, dLdw: Weights) -> tuple[Self, Weights]:
        assert (
            w.layers()
            == dLdw.layers()
            == self.last_update.layers()
            == self.actual.layers()
        ), f"{w.layers()} == {dLdw.layers()} == {self.last_update.layers()} == {self.actual.layers()}"
        last = dLdw.combine(
            self.last_update,
            lambda di, lu: self.lr * di - self.momentum * lu,
            lambda di, lu: self.lr * di - self.momentum * lu,
        )
        updated = w.combine(last, lambda wi, lu: wi - lu, lambda bi, lu: bi - lu)
        return (
            self._replace(last_update=last, actual=updated),
            updated.combine(
                self.last_update,
                lambda wi, lu: wi + self.overstep * lu,
                lambda bi, lu: bi + self.overstep * lu,
            ),
        )


@jaxtyped(typechecker=beartype)
class RMSProp(NamedTuple):
    lr: Float[Array, ""]

    moving_decay: Float[Array, ""]
    moving_average: Weights

    @jit
    def __call__(self, w: Weights, dLdw: Weights) -> tuple[Self, Weights]:
        assert (
            w.layers() == dLdw.layers() == self.moving_average.layers()
        ), f"{w.layers()} == {dLdw.layers()} == {self.moving_average.layers()}"
        squared = dLdw.map(jnp.square, jnp.square)
        persistent = self.moving_average.map(
            lambda w: self.moving_decay * w,
            lambda b: self.moving_decay * b,
        )
        novel = squared.map(
            lambda w: (1.0 - self.moving_decay) * w,
            lambda b: (1.0 - self.moving_decay) * b,
        )
        moving_avg = persistent.combine(
            novel,
            lambda wi, ni: wi + ni,
            lambda bi, ni: bi + ni,
        )
        rms = moving_avg.map(jnp.sqrt, jnp.sqrt)
        update = dLdw.combine(
            rms,
            lambda di, ri: self.lr * di / ri,
            lambda di, ri: self.lr * di / ri,
        )
        updated = w.combine(update, lambda wi, ui: wi - ui, lambda bi, ui: bi - ui)
        return self._replace(moving_average=moving_avg), updated
