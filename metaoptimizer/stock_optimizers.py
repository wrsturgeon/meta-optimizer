from metaoptimizer.nontest import jit
from metaoptimizer.weights import Weights

from beartype import beartype
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float
from typing import NamedTuple


@jaxtyped(typechecker=beartype)
class SGD(NamedTuple):
    lr: jnp.float32

    @jit
    def __call__(self, w: Weights, dLdw: Weights) -> Weights:
        assert w.layers() == dLdw.layers()
        return Weights(
            [wi - self.lr * di for wi, di in zip(w.W, dLdw.W)],
            [bi - self.lr * di for bi, di in zip(w.B, dLdw.B)],
        )
