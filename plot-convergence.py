from metaoptimizer import feedforward, trial

from jax import nn as jnn, numpy as jnp, random as jrnd
import os


DIRECTORY = ["convergence-rates"]

TRIALS = 32
NONLINEARITY = jnn.gelu
POWER = jnp.array(2.0, dtype=jnp.float32)
TRAINING_STEPS = 1000
from metaoptimizer.optimizers import (
    sgd as optim,
    # adam as optim,
)

forward_pass = lambda weights, x: feedforward.run(weights, x, NONLINEARITY)
optimizer = optim.update
opt_params = optim.defaults()


if __name__ == "__main__":

    print("And so it begins...")

    for layers in range(1, 4):
        print(f"  {layers} layer" + ("" if layers == 1 else "s"))
        layer_dir = (*DIRECTORY, f"{layers}-layer")

        for lg_ndim in range(4):
            ndim = int(jnp.exp2(lg_ndim))
            print(f"    {ndim} dimension" + ("" if ndim == 1 else "s"))
            ndim_dir = (*layer_dir, f"{ndim}-dimensional")

            shapes = tuple([ndim for _ in range(layers + 1)])
            w_example = feedforward.init(shapes, jrnd.PRNGKey(42), False)
            opt_state = optim.init(w_example, opt_params)

            # for lg_lr in range(8):
            #     lr = 0.0001 * jnp.exp2(lg_lr)
            #     print(f"      learning rate of {lr}")
            #     lr_dir = (*ndim_dir, f"{lr}-learning-rate")

            for lg_dist in range(8):
                dist = 0.01 * jnp.exp2(lg_dist)
                print(f"        initial distance of {dist}")
                dist_dir = (*ndim_dir, f"{dist}-distance")

                # for lg_batch in range(4):
                #     batch = int(jnp.exp2(lg_batch))
                #     print(f"          batches of {batch}")
                #     batch_dir = (*dist_dir, f"batch-of-{batch}")
                batch = 1

                for i in range(TRIALS):
                    directory = (*dist_dir, f"trial-{i}")

                    if not os.path.exists(os.path.join(*directory)):
                        print(f"            trial #{i}")
                        trial.run(
                            jrnd.PRNGKey(i),
                            ndim,
                            batch,
                            layers,
                            forward_pass,
                            optimizer,
                            opt_state,
                            opt_params,
                            NONLINEARITY,
                            TRAINING_STEPS,
                            directory,
                            POWER,
                            dist,
                            "              ",
                            verbose=False,
                        )

    import plot  # Relative import: `plot.py`

    plot.run(os.path.join(os.getcwd(), *DIRECTORY))
