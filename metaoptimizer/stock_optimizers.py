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
        return self, Weights(
            [wi - self.lr * di for wi, di in zip(w.W, dLdw.W)],
            [bi - self.lr * di for bi, di in zip(w.B, dLdw.B)],
        )


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
        last_w = [
            self.lr * di - self.momentum * lu
            for di, lu in zip(dLdw.W, self.last_update.W)
        ]
        last_b = [
            self.lr * di - self.momentum * lu
            for di, lu in zip(dLdw.B, self.last_update.B)
        ]
        return self._replace(last_update=Weights(last_w, last_b)), Weights(
            [wi - lu for wi, lu in zip(w.W, last_w)],
            [bi - lu for bi, lu in zip(w.B, last_b)],
        )
