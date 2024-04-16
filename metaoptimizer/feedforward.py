from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Callable, Tuple
from functools import partial
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax.experimental.checkify import check
from jax.numpy import linalg as jla
from jax_dataclasses import Static
from jaxtyping import jaxtyped, Array, Float32, Float64, PyTree, UInt
import os


if os.getenv("NONJIT") == "1":
    print("NOTE: `NONJIT` activated")
else:
    from jax.experimental.checkify import all_checks, checkify, Error
    from jax_dataclasses import jit


KeyArray = UInt[Array, "n_keys"]  # <https://github.com/google/jax/issues/12706>


@jaxtyped(typechecker=beartype)
def raw_nonlinear(
    w: Float64[Array, "z_out z_in"],
    b: Float64[Array, "z_out"],
    x: Float32[Array, "batch z_in 1"],
    nl: Static[
        Callable[
            [Float32[Array, "batch z_out 1"]],
            Float32[Array, "batch z_out 1"],
        ]
    ],
) -> Float32[Array, "batch z_out 1"]:
    return nl(w.astype(jnp.float32) @ x + b.astype(jnp.float32))


if os.getenv("NONJIT") == "1":

    def nonlinear(
        w: Float64[Array, "z_out z_in"],
        b: Float64[Array, "z_out"],
        x: Float32[Array, "batch z_in 1"],
        nl: Static[
            Callable[
                [Float32[Array, "batch z_out 1"]],
                Float32[Array, "batch z_out 1"],
            ]
        ],
    ) -> Float32[Array, "batch z_out 1"]:
        return raw_nonlinear(w, b, x, nl)

else:

    @jit
    def jit_nonlinear(
        w: Float64[Array, "z_out z_in"],
        b: Float64[Array, "z_out"],
        x: Float32[Array, "batch z_in 1"],
        nl: Static[
            Callable[
                [Float32[Array, "batch z_out 1"]],
                Float32[Array, "batch z_out 1"],
            ]
        ],
    ) -> Tuple[Error, Float32[Array, "batch z_out 1"]]:
        return checkify(raw_nonlinear)(w, b, x, nl)

    def nonlinear(
        w: Float64[Array, "z_out z_in"],
        b: Float64[Array, "z_out"],
        x: Float32[Array, "batch z_in 1"],
        nl: Static[
            Callable[
                [Float32[Array, "batch z_out 1"]],
                Float32[Array, "batch z_out 1"],
            ]
        ],
    ) -> Float32[Array, "batch z_out 1"]:
        err, y = jit_nonlinear(w, b, x, nl)
        err.throw()
        return y


# TODO: What is wrong with the typechecker and `Weights` here?!?
@jaxtyped(typechecker=beartype)
def raw_run(
    weights: Weights,
    x: Float32[Array, "batch n_in"],
    nl: Static[Callable[[Float32[Array, "..."]], Float32[Array, "..."]]] = jnn.gelu,
) -> Float32[Array, "batch n_out"]:
    batch, ndim_in = x.shape
    x = x[..., jnp.newaxis]
    # check(
    #     jnp.all(jnp.isfinite(x)),
    #     """
    #     `feedforward` got an input `x` with non-finite values.
    #     Original `x` was
    #     {x}
    #     """,
    #     x=x,
    # )
    n = weights.layers()
    for w, b in zip(weights.W, weights.B):
        x = nonlinear(w, b, x, nl)
        # check(
        #     jnp.all(jnp.isfinite(y)),
        #     """
        #     `feedforward` produced a hidden `x` with non-finite values (after layer #{i}).
        #     Original `x` was
        #     {x}
        #     `w.W[i]` was
        #     {w}
        #     `w.B[i]` was
        #     {b}
        #     After, `x` was
        #     {y}
        #     """,
        #     i=jnp.array(i, dtype=jnp.uint32),
        #     x=x,
        #     w=w.W[i],
        #     b=w.B[i],
        #     y=y,
        # )
        # x = y
        # del y
    assert x.shape[-1] == 1
    return x[..., 0]


if os.getenv("NONJIT") == "1":

    def run(
        w: Weights,
        x: Float32[Array, "batch n_in"],
        nl: Static[Callable[[Float32[Array, "..."]], Float32[Array, "..."]]] = jnn.gelu,
    ) -> Float32[Array, "batch n_out"]:
        return raw_run(w, x, nl)

else:

    @jit
    def jit_run(
        w: Weights,
        x: Float32[Array, "batch n_in"],
        nl: Static[Callable[[Float32[Array, "..."]], Float32[Array, "..."]]] = jnn.gelu,
    ) -> Tuple[Error, Float32[Array, "batch n_out"]]:
        return checkify(raw_run, errors=all_checks)(w, x, nl)

    def run(
        w: Weights,
        x: Float32[Array, "batch n_in"],
        nl: Static[Callable[[Float32[Array, "..."]], Float32[Array, "..."]]] = jnn.gelu,
    ) -> Float32[Array, "batch n_out"]:
        err, y = jit_run(w, x, nl)
        err.throw()
        return y


@jaxtyped(typechecker=beartype)
def init(sizes: list[int], key: KeyArray, random_biases: bool) -> Weights:
    n = len(sizes)
    W = []
    B = []
    he = jnn.initializers.he_normal(dtype=jnp.float64)
    for i in range(1, n):
        key, k = jrnd.split(key)
        if random_biases:
            k, k1 = jrnd.split(k)
            B.append(jrnd.normal(k1, [sizes[i]], dtype=jnp.float64))
        else:
            B.append(jnp.zeros(sizes[i], dtype=jnp.float64))
        W.append(he(k, [sizes[i], sizes[i - 1]], dtype=jnp.float64))
    return Weights(W=W, B=B)
