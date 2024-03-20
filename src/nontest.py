import jax
import sys
from typing import Callable


def nontest(f: Callable) -> Callable:
    """
    Execute a function on iff NOT running `pytest`.
    Otherwise, return the identity function (pass input unchanged).
    """
    if "pytest" not in sys.modules:
        return f
    return lambda x: x


def jit(f, **kwargs):
    if "pytest" not in sys.modules:
        return jax.jit(f, **kwargs)
    return f
