from metaoptimizer.weights import Weights

from beartype.typing import Callable, Tuple
from jaxtyping import Array, Float, PyTree


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
