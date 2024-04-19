# Utilities for selectively enabling JIT compilation.


from beartype import beartype
from beartype.typing import Callable
from jaxtyping import jaxtyped
import os


print(f"Environment variable `NONJIT` is `{os.getenv('NONJIT')}`")
if os.getenv("NONJIT") == "1":

    print("*** NOTE: `NONJIT` enabled")

    def jit(*static_argnums) -> Callable[[Callable], Callable]:
        return jaxtyped(typechecker=beartype)  # itself a function

else:

    print("*** NOTE: `NONJIT` NOT enabled; JIT-compiling everything...")

    from jax import jit as jax_jit
    from jax.experimental.checkify import checkify, all_checks

    def jit(*static_argnums) -> Callable[[Callable], Callable]:
        def partially_applied(f: Callable) -> Callable:
            f_err = jax_jit(
                checkify(
                    jaxtyped(f, typechecker=beartype),
                    errors=all_checks,
                ),
                static_argnums=static_argnums,
            )

            def handle_err(*args, **kwargs):
                err, y = f_err(*args, **kwargs)
                err.throw()
                return y

            return handle_err

        return partially_applied
