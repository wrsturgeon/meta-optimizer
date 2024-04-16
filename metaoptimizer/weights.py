from beartype.typing import Callable, List, Self
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float64
from typing import NamedTuple


@pytree_dataclass
class Weights:
    W: List[Float64[Array, "..."]]
    B: List[Float64[Array, "..."]]

    def layers(self: Self) -> int:
        n = len(self.W)
        assert len(self.B) == n
        return n
