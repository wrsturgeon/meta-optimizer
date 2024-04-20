from metaoptimizer import permutations
from metaoptimizer.jit import jit
from metaoptimizer.optimizers import Optimizer
from metaoptimizer.permutations import Permutation

from beartype import beartype
from beartype.typing import Any, Callable, List, Tuple
from jax import grad, numpy as jnp, value_and_grad
from jax.errors import TracerBoolConversionError
from jax.lax import stop_gradient
from jax.tree_util import tree_map, tree_reduce, tree_structure
from jaxtyping import jaxtyped, Array, Float32, Float64, PyTree
import operator
import os
import sys


ForwardPass = Callable[
    [PyTree[Float64[Array, "..."]], Float32[Array, "batch ndim_in"]],
    Float32[Array, "batch ndim_out"],
]


OPTIMIZER_LR = jnp.array(0.25, dtype=jnp.float64)


@jit(1)
def loss(
    weights: PyTree[Float64[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float32[Array, "batch ndim_in"],
    ground_truth: Float32[Array, "batch ndim_out"],
    power: Float32[Array, ""] = jnp.array(2, dtype=jnp.float32),
) -> Float32[Array, ""]:
    outputs = forward_pass(weights, inputs)
    assert (
        outputs.shape == ground_truth.shape
    ), f"{outputs.shape} =/= {ground_truth.shape}"
    L1 = jnp.abs(ground_truth - outputs)
    Ln = jnp.pow(L1, power)
    return jnp.sum(Ln)


loss_and_grad = value_and_grad(loss)


@jit(1, 4)
def step(
    weights: PyTree[Float64[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float32[Array, "batch ndim_in"],
    ground_truth: Float32[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float64[Array, ""]],
    opt_state: PyTree[Float64[Array, "..."]],
    power: Float32[Array, ""] = jnp.array(2, dtype=jnp.float32),
) -> Tuple[
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, "..."]],
    Float32[Array, ""],
]:
    L, dLdw = loss_and_grad(weights, forward_pass, inputs, ground_truth, power)
    opt_state_adjusted, weights_adjusted = optim_parameterized(
        opt_params, opt_state, weights, dLdw
    )
    return weights_adjusted, opt_state_adjusted, L


@jit(1, 4)
def update_and_retest(
    weights: PyTree[Float64[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float32[Array, "batch ndim_in"],
    ground_truth: Float32[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float64[Array, ""]],
    opt_state: PyTree[Float64[Array, "..."]],
    last_dLdw: PyTree[Float64[Array, "..."]],
    power: Float32[Array, ""] = jnp.array(2, dtype=jnp.float32),
) -> Tuple[
    Float32[Array, ""],
    Tuple[PyTree[Float64[Array, "..."]], PyTree[Float64[Array, "..."]]],
]:
    opt_state_adjusted, weights_adjusted = optim_parameterized(
        opt_params, opt_state, weights, last_dLdw
    )
    return loss(weights, forward_pass, inputs, ground_truth, power), (
        weights_adjusted,
        opt_state_adjusted,
    )


@jit(2)
def slope_away_from_local_minimum(
    opt_params: PyTree[Float64[Array, ""]],
    opt_state: PyTree[Float64[Array, "..."]],
    optim_parameterized: Optimizer,
    weights: PyTree[Float64[Array, "..."]],
    dLdw: PyTree[Float64[Array, "..."]],
) -> Float64[Array, ""]:
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


@jit(1, 4)
def step_downhill(
    weights: PyTree[Float64[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float32[Array, "batch ndim_in"],
    ground_truth: Float32[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float64[Array, ""]],
    opt_state: PyTree[Float64[Array, "..."]],
    last_dLdw: PyTree[Float64[Array, "..."]],
    power: Float32[Array, ""] = jnp.array(2, dtype=jnp.float32),
) -> Tuple[
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, ""]],
    Float32[Array, ""],
    PyTree[Float64[Array, "..."]],
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
    opt_params_adjusted: PyTree[Float64[Array, "..."]] = tree_map(
        lambda w, d: w - OPTIMIZER_LR * d,
        opt_params,
        dLdo,
    )
    return (
        weights_adjusted,
        opt_state_adjusted,
        opt_params_adjusted,
        L,
        dLdw,
    )


@jit(2)
# @jaxtyped(typechecker=beartype)
def opt_step_global(
    opt_params: PyTree[Float64[Array, ""]],
    opt_state: PyTree[Float64[Array, "..."]],
    optim_parameterized: Optimizer,
    weights: PyTree[Float64[Array, "..."]],
    dLdw: PyTree[Float64[Array, "..."]],
    global_minimum: PyTree[Float64[Array, "..."]],
) -> Tuple[
    Float32[Array, ""],
    Tuple[
        PyTree[Float64[Array, "..."]],
        PyTree[Float64[Array, "..."]],
        List[Permutation],
    ],
]:
    # try:
    #     assert tree_reduce(
    #         operator.and_,
    #         tree_map(
    #             lambda x: jnp.all(jnp.isfinite(x)),
    #             weights,
    #         ),
    #     ), "Non-finite input"
    # except TracerBoolConversionError:
    #     sys.exit("Please don't JIT-compile `training.opt_step_global`!")

    print("Calling `optim_parameterized`...")
    opt_state_adjusted, weights_adjusted = jit()(optim_parameterized)(
        opt_params, opt_state, weights, dLdw
    )
    print("Calling `permutations.layer_distance`...")
    L, perm = permutations.layer_distance(
        actual=weights_adjusted,
        ideal=global_minimum,
    )
    print(f"Compiling `opt_step_global`...")
    return L, (opt_state_adjusted, weights_adjusted, perm)


@jit(1, 4)
# @jaxtyped(typechecker=beartype)
def step_global(
    weights: PyTree[Float64[Array, "..."]],
    forward_pass: ForwardPass,
    inputs: Float32[Array, "batch ndim_in"],
    ground_truth: Float32[Array, "batch ndim_out"],
    optim_parameterized: Optimizer,
    opt_params: PyTree[Float64[Array, ""]],
    opt_state: PyTree[Float64[Array, "..."]],
    global_minimum: PyTree[Float64[Array, "..."]],
    power: Float32[Array, ""] = jnp.array(2, dtype=jnp.float32),
) -> Tuple[
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, ""]],
    List[Permutation],
    Float32[Array, ""],
]:
    print("Calling `loss_and_grad`...")
    L, dLdw = loss_and_grad(weights, forward_pass, inputs, ground_truth, power)
    print("Calling `grad`...")
    dLdo, (opt_state_adjusted, weights_adjusted, perm) = grad(
        opt_step_global,
        has_aux=True,
    )(
        opt_params,
        opt_state,
        optim_parameterized,
        weights,
        dLdw,
        global_minimum,
    )
    print("Stepping parameters...")
    opt_params_adjusted: PyTree[Float64[Array, "..."]] = tree_map(
        lambda w, d: w - OPTIMIZER_LR * d,
        opt_params,
        dLdo,
    )
    print(f"Compiling `step_global`...")
    return (
        weights_adjusted,
        opt_state_adjusted,
        opt_params_adjusted,
        perm,
        L,
    )
