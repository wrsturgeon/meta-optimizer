from metaoptimizer.nontest import jit

from beartype import beartype
from jaxtyping import jaxtyped, Array, Float
from typing import NamedTuple


@jaxtyped(typechecker=beartype)
class Weights(NamedTuple):
    W: list[Float[Array, "..."]]
    B: list[Float[Array, "..."]]

    @jit
    def layers(self) -> int:
        n = len(self.W)
        assert len(self.B) == n
        return n
