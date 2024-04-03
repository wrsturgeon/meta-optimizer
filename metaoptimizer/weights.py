from beartype import beartype
from beartype.typing import Callable, Self
from jaxtyping import jaxtyped, Array, Float
from typing import NamedTuple


@jaxtyped(typechecker=beartype)
class Weights(NamedTuple):
    W: list[Float[Array, "..."]]
    B: list[Float[Array, "..."]]

    def layers(self: Self) -> int:
        n = len(self.W)
        assert len(self.B) == n
        return n
