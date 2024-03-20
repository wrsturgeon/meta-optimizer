from beartype import beartype
from beartype.typing import Callable
import jax
import sys


@beartype
def nontest(f: Callable) -> Callable:
    """
    Execute a function on iff NOT running `pytest`.
    Otherwise, return the identity function (pass input unchanged).
    """
    if "pytest" not in sys.modules:
        return f
    return lambda x: x


@beartype
def jit(f: Callable, **kwargs):
    if "pytest" not in sys.modules:
        return jax.jit(f, **kwargs)
    return f
