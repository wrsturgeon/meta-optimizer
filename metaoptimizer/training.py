from metaoptimizer.optimizers import Optimizer
from metaoptimizer.weights import Weights

from beartype import beartype
from beartype.typing import Any, Callable, Tuple
from functools import partial
from jax import jit, numpy as jnp, value_and_grad
from jax.experimental.checkify import all_checks, checkify
from jaxtyping import jaxtyped, Array, Float, PyTree


ForwardPass = Callable[
    [Weights, Float[Array, "batch ndim_in"]], Float[Array, "batch ndim_out"]
]


@jaxtyped(typechecker=beartype)
def loss(
    forward_pass: ForwardPass,
    weights: Weights,
    inputs: Float[Array, "batch ndim_in"],
    ground_truth: Float[Array, "batch ndim_out"],
    power: Float[Array, ""] = jnp.ones([]),
) -> Float[Array, ""]:
    outputs = forward_pass(weights, inputs)
    assert isinstance(outputs, jnp.ndarray), f"`{outputs}` is not a JAX array"
    assert (
        outputs.shape == ground_truth.shape
    ), f"{outputs.shape} =/= {ground_truth.shape}"
    L1 = jnp.abs(ground_truth - outputs)
    Ln = jnp.pow(L1, power)
    return jnp.sum(Ln)


@partial(jit, static_argnums=[0, 4])
@partial(checkify, errors=all_checks)
@jaxtyped(typechecker=beartype)
def step(
    forward_pass: ForwardPass,
    weights: Weights,
    inputs: Float[Array, "batch ndim_in"],
    ground_truth: Float[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float[Array, "..."]],
    opt_state: PyTree[Float[Array, "..."]],
    power: Float[Array, ""] = jnp.ones([]),
) -> Tuple[
    PyTree[Float[Array, "..."]],
    Weights,
    Float[Array, ""],
]:
    L, dLdw = value_and_grad(loss, argnums=1)(
        forward_pass, weights, inputs, ground_truth, power
    )
    opt_state_adjusted, weights_adjusted = optim_parameterized(
        opt_params, opt_state, weights, dLdw
    )
    return opt_state_adjusted, weights_adjusted, L
