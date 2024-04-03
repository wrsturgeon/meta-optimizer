from metaoptimizer.optimizers import Optimizer

from beartype import beartype
from beartype.typing import Any, Callable, Tuple
from functools import partial
from jax import grad, jit, numpy as jnp, value_and_grad
from jax.lax import stop_gradient
from jax.tree_util import tree_map, tree_reduce, tree_structure
from jax.experimental.checkify import all_checks, checkify
from jaxtyping import jaxtyped, Array, Float, PyTree
import operator


ForwardPass = Callable[
    [PyTree[Float[Array, "..."]], Float[Array, "batch ndim_in"]],
    Float[Array, "batch ndim_out"],
]


@jaxtyped(typechecker=beartype)
def loss(
    weights: PyTree[Float[Array, "..."]],
    forward_pass: ForwardPass,
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


loss_and_grad = value_and_grad(loss)


@partial(jit, static_argnums=[1, 4])
@partial(checkify, errors=all_checks)
@jaxtyped(typechecker=beartype)
def step(
    weights: PyTree[Float[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float[Array, "batch ndim_in"],
    ground_truth: Float[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float[Array, ""]],
    opt_state: PyTree[Float[Array, "..."]],
    power: Float[Array, ""] = jnp.ones([]),
) -> Tuple[
    PyTree[Float[Array, "..."]],
    PyTree[Float[Array, "..."]],
    Float[Array, ""],
]:
    L, dLdw = loss_and_grad(weights, forward_pass, inputs, ground_truth, power)
    opt_state_adjusted, weights_adjusted = optim_parameterized(
        opt_params, opt_state, weights, dLdw
    )
    return weights_adjusted, opt_state_adjusted, L


@jaxtyped(typechecker=beartype)
def update_and_retest(
    weights: PyTree[Float[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float[Array, "batch ndim_in"],
    ground_truth: Float[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float[Array, ""]],
    opt_state: PyTree[Float[Array, "..."]],
    last_dLdw: PyTree[Float[Array, "..."]],
    power: Float[Array, ""] = jnp.ones([]),
) -> Tuple[
    Float[Array, ""], Tuple[PyTree[Float[Array, "..."]], PyTree[Float[Array, "..."]]]
]:
    opt_state_adjusted, weights_adjusted = optim_parameterized(
        opt_params, opt_state, weights, last_dLdw
    )
    return loss(weights, forward_pass, inputs, ground_truth, power), (
        weights_adjusted,
        opt_state_adjusted,
    )


@jaxtyped(typechecker=beartype)
def slope_away_from_local_minimum(
    opt_params: PyTree[Float[Array, ""]],
    opt_state: PyTree[Float[Array, "..."]],
    optim_parameterized: Optimizer,
    weights: PyTree[Float[Array, "..."]],
    dLdw: PyTree[Float[Array, "..."]],
) -> Float[Array, ""]:
    # TODO: directly do the math instead of recomputing,
    # but make sure it's right (at least here I'm sure)
    # ALL THIS IS REALLY DOING IS MINIMIZING `dLdw` BY MOVING `weights`
    _, actual = optim_parameterized(opt_params, opt_state, weights, dLdw)
    forgotten = stop_gradient(actual)
    downhill = tree_map(operator.sub, forgotten, dLdw)
    return tree_reduce(
        operator.add,
        tree_map(lambda a, b: jnp.sum(jnp.abs(a - b)), downhill, actual),
    )


@partial(jit, static_argnums=[1, 4])
@partial(checkify, errors=all_checks)
@jaxtyped(typechecker=beartype)
def step_combined(
    weights: PyTree[Float[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float[Array, "batch ndim_in"],
    ground_truth: Float[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float[Array, ""]],
    opt_state: PyTree[Float[Array, "..."]],
    meta_opt_state: PyTree[Float[Array, "..."]],
    last_dLdw: PyTree[Float[Array, "..."]],
    power: Float[Array, ""] = jnp.ones([]),
) -> Tuple[
    PyTree[Float[Array, "..."]],
    PyTree[Float[Array, "..."]],
    PyTree[Float[Array, ""]],
    PyTree[Float[Array, "..."]],
    Float[Array, ""],
    PyTree[Float[Array, "..."]],
]:
    # TODO: This loss function probably won't make sense for Nesterov momentum,
    # since it makes no distinction between actual weights and returned weights

    (L, (weights_adjusted, opt_state_adjusted)), dLdw = value_and_grad(
        update_and_retest, has_aux=True
    )(
        weights,
        forward_pass,
        inputs,
        ground_truth,
        optim_parameterized,
        opt_params,
        opt_state,
        last_dLdw,
        power,
    )
    dLdo = grad(slope_away_from_local_minimum)(
        opt_params,
        opt_state,
        optim_parameterized,
        weights,
        dLdw,
    )

    # TODO: REINSTATE
    # meta_opt_state_adjusted, opt_params_adjusted = optim_parameterized(
    #     opt_params, meta_opt_state, opt_params, dLdo
    # )
    opt_params_adjusted = tree_map(lambda w, d: w - 0.01 * d, opt_params, dLdo)
    meta_opt_state_adjusted = meta_opt_state

    return (
        weights_adjusted,
        opt_state_adjusted,
        opt_params_adjusted,
        meta_opt_state_adjusted,
        L,
        dLdw,
    )
