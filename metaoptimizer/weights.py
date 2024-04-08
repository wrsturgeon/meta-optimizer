from beartype.typing import Callable, List, Self
from jaxtyping import Array, Float64
from typing import NamedTuple


class Weights(NamedTuple):
    W: List[Float64[Array, "..."]]
    B: List[Float64[Array, "..."]]

    def layers(self: Self) -> int:
        n = len(self.W)
        assert len(self.B) == n
        return n
