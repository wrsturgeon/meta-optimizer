from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import jaxtyped, Array, Float, PyTree


# TODO: Why don't named annotations like "P" & "S" work here?
Optimizer = Callable[
    [
        PyTree[Float[Array, "..."]],
        PyTree[Float[Array, "..."]],
        Weights,
        Weights,
    ],
    Tuple[PyTree[Float[Array, "..."]], Weights],
]


@jaxtyped(typechecker=beartype)
def typecheck(_: Optimizer) -> None:
    pass
