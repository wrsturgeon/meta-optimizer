from beartype import beartype
from matplotlib import pyplot as plt
from jaxtyping import jaxtyped
import numpy as np
import os


DPI = 256
LINE_WIDTH = DPI / 256.0


@jaxtyped(typechecker=beartype)
def run(directory: str) -> None:

    if not os.path.exists(directory):
        return
    assert os.path.isdir(directory), f"File `{directory}` is not a directory"

    plt.close("all")

    # If we have historical data, make sure axis ranges are identical
    historical_files, non_historical_files = [], []
    for f in os.listdir(directory):
        if f.endswith("historical.npy"):
            historical_files.append(f)
        else:
            non_historical_files.append(f)
    historical_loaded = [np.load(os.path.join(directory, f)) for f in historical_files]
    n_historical = len(historical_files)
    if n_historical != 0:
        historical_min = np.min([np.min(f) for f in historical_loaded])
        historical_max = np.max([np.max(f) for f in historical_loaded])
        historical_range = historical_max - historical_min
    for fname, arr in zip(historical_files, historical_loaded):
        plt.plot(arr, linewidth=LINE_WIDTH)
        plt.ylim(
            historical_min - 0.1 * historical_range,
            historical_max + 0.1 * historical_range,
        )
        plt.ticklabel_format(style="plain", useOffset=False)
        os.makedirs(os.path.join(directory, "plots"), exist_ok=True)
        without_ext, ext = os.path.splitext(fname)
        plt.savefig(
            os.path.join(
                directory,
                "plots",
                without_ext + ".png",
            ),
            dpi=DPI,
        )
        plt.close()

    for fname in non_historical_files:
        f = os.path.join(directory, fname)
        # checking if it is a file
        if os.path.isdir(f):
            run(f)
        elif os.path.isfile(f):
            without_ext, ext = os.path.splitext(fname)
            if ext == ".npy":
                plt.plot(np.load(f), linewidth=LINE_WIDTH)
                # plt.gca().set_ylim([0.0, 1.0])
                plt.ticklabel_format(style="plain", useOffset=False)
                os.makedirs(os.path.join(directory, "plots"), exist_ok=True)
                plt.savefig(
                    os.path.join(
                        directory,
                        "plots",
                        without_ext + ".png",
                    ),
                    dpi=DPI,
                )
                plt.close()
            else:
                print(f"Skipping {f} (extension was `{ext}` instead of `.npy`)")
        else:
            print(f"Skipping {f} (not a file or directory)")


if __name__ == "__main__":
    run(os.path.join(os.getcwd(), "logs"))
    run(os.path.join(os.getcwd(), "convergence-rates"))
