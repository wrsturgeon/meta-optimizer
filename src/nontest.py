from beartype import beartype
from beartype.typing import Callable
import jax
import sys


@beartype
def jit(f: Callable, **kwargs):
    if "pytest" not in sys.modules:  # pragma: no cover
        return jax.jit(f, **kwargs)
    return f
