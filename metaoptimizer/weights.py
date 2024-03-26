from metaoptimizer.nontest import jit

from beartype import beartype
from beartype.typing import Callable, Self
from jaxtyping import jaxtyped, Array, Float
from typing import NamedTuple


@jaxtyped(typechecker=beartype)
class Weights(NamedTuple):
    W: list[Float[Array, "..."]]
    B: list[Float[Array, "..."]]

    @jit
    def layers(self: Self) -> int:
        n = len(self.W)
        assert len(self.B) == n
        return n

    @jit
    def map(
        self: Self,
        fw: Callable[[Float[Array, "..."]], Float[Array, "..."]],
        fb: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    ) -> Self:
        return self._replace(W=[fw(w) for w in self.W], B=[fb(b) for b in self.B])

    @jit
    def combine(
        self: Self,
        other: Self,
        fw: Callable[[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]],
        fb: Callable[[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]],
    ) -> Self:
        return self._replace(
            W=[fw(a, b) for a, b in zip(self.W, other.W)],
            B=[fb(a, b) for a, b in zip(self.B, other.B)],
        )
