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
        return Weights(w.W - self.lr * dLdw.W, w.B - self.lr * dLdw.B)
