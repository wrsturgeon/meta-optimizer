from metaoptimizer import trial

from beartype import beartype
from beartype.typing import Callable, List, Tuple
from functools import partial
from importlib import import_module
from jax import nn as jnn, numpy as jnp, random as jrnd
from jax_dataclasses import Static
from jaxtyping import jaxtyped, Array, Float32, Float64
import os
from shutil import rmtree
from types import ModuleType


if os.getenv("NONJIT") == "1":
    print("NOTE: `NONJIT` activated")
else:
    from jax.experimental.checkify import all_checks, checkify, Error
    from jax_dataclasses import jit


DIRECTORY = ["convergence-rates"]

TRIALS = 100
NONLINEARITY = jnn.gelu
POWER = jnp.array(2.0, dtype=jnp.float32)
TRAINING_STEPS = 1000
from metaoptimizer.optimizers import (
    # sgd as OPTIM,
    adam as OPTIM,
)


if os.getenv("NONJIT") == "1":

    def run(
        key: Array,
        ndim: Static[int] = 3,
        batch: Static[int] = 1,
        layers: Static[int] = 2,
        nonlinearity: Static[
            Callable[[Float32[Array, "*n"]], Float32[Array, "*n"]]
        ] = jnn.gelu,
        optim: Static[ModuleType] = import_module("metaoptimizer.optimizers.sgd"),
        training_steps: Static[int] = 100000,
        subdir: Static[List[str]] = [],
        convergence_only: Static[bool] = False,
        power: Float32[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
        lr: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
        initial_distance: Float64[Array, ""] = jnp.array(0.1, dtype=jnp.float64),
    ) -> None:
        return trial.run(
            key,
            ndim,
            batch,
            layers,
            nonlinearity,
            optim,
            training_steps,
            list(subdir),
            convergence_only,
            power,
            lr,
            initial_distance,
        )

else:

    @jit
    def jit_run(
        key: Array,
        ndim: Static[int] = 3,
        batch: Static[int] = 1,
        layers: Static[int] = 2,
        nonlinearity: Static[
            Callable[[Float32[Array, "*n"]], Float32[Array, "*n"]]
        ] = jnn.gelu,
        optim: Static[ModuleType] = import_module("metaoptimizer.optimizers.sgd"),
        training_steps: Static[int] = 100000,
        subdir: Static[Tuple] = (),
        convergence_only: Static[bool] = False,
        power: Float32[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
        lr: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
        initial_distance: Float64[Array, ""] = jnp.array(0.1, dtype=jnp.float64),
    ) -> Tuple[Error, None]:
        return checkify(trial.run, errors=all_checks)(
            key,
            ndim,
            batch,
            layers,
            nonlinearity,
            optim,
            training_steps,
            list(subdir),
            convergence_only,
            power,
            lr,
            initial_distance,
        )

    def run(
        key: Array,
        ndim: Static[int] = 3,
        batch: Static[int] = 1,
        layers: Static[int] = 2,
        nonlinearity: Static[
            Callable[[Float32[Array, "*n"]], Float32[Array, "*n"]]
        ] = jnn.gelu,
        optim: Static[ModuleType] = import_module("metaoptimizer.optimizers.sgd"),
        training_steps: Static[int] = 100000,
        subdir: Static[List[str]] = [],
        convergence_only: Static[bool] = False,
        power: Float32[Array, ""] = jnp.array(2.0, dtype=jnp.float32),
        lr: Float64[Array, ""] = jnp.array(0.01, dtype=jnp.float64),
        initial_distance: Float64[Array, ""] = jnp.array(0.1, dtype=jnp.float64),
    ) -> None:
        err, y = jit_run(
            key,
            ndim,
            batch,
            layers,
            nonlinearity,
            optim,
            training_steps,
            tuple(subdir),
            convergence_only,
            power,
            lr,
            initial_distance,
        )
        err.throw()
        return y


if __name__ == "__main__":

    print("And so it begins...")
    for layers in range(1, 4):
        print(f"  {layers} layer" + ("" if layers == 0 else "s"))
        layer_dir = [*DIRECTORY, f"{layers}-layer"]
        for lg_ndim in range(4):
            ndim = int(jnp.exp2(lg_ndim))
            print(f"    {ndim} dimension" + ("" if ndim == 0 else "s"))
            ndim_dir = [*layer_dir, f"{ndim}-dimensional"]
            for lg_lr in range(8):
                lr = 0.0001 * jnp.exp2(lg_lr)
                print(f"      learning rate of {lr}")
                lr_dir = [*ndim_dir, f"{lr}-learning-rate"]
                for lg_dist in range(8):
                    dist = 0.01 * jnp.exp2(lg_dist)
                    print(f"        initial distance of {dist}")
                    dist_dir = [*lr_dir, f"{dist}-distance"]
                    for lg_batch in range(8):
                        batch = int(jnp.exp2(lg_batch))
                        print(f"          batches of {batch}")
                        batch_dir = [*dist_dir, f"batch-of-{batch}"]
                        for i in range(TRIALS):
                            print(f"            trial #{i}")
                            trial_dir = [*batch_dir, f"trial-{i}"]

                            run(
                                jrnd.PRNGKey(i),
                                ndim,
                                batch,
                                layers,
                                NONLINEARITY,
                                OPTIM,
                                TRAINING_STEPS,
                                trial_dir,
                                True,
                                POWER,
                                lr,
                                dist,
                            )

    import plot  # Relative import: `plot.py`

    plot.run()
