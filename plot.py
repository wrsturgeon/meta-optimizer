from matplotlib import pyplot as plt
import numpy as np
import os


DPI = 256


def run(directory: str = os.path.join(os.getcwd(), "logs")) -> None:
    assert os.path.exists(
        directory
    ), f"Directory `{directory}` does not exist. Try running the training loop first."
    assert os.path.isdir(directory), f"File `{directory}` is not a directory"

    plt.close("all")
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isdir(f):
            run(f)
        elif os.path.isfile(f):
            without_ext, ext = os.path.splitext(f)
            if ext == ".npy":
                plt.plot(np.load(f))
                # plt.gca().set_ylim([0.0, 1.0])
                plt.ticklabel_format(style="plain", useOffset=False)
                plt.savefig(without_ext + ".png", dpi=DPI)
                plt.close()
            else:
                print(f"Skipping {f} (extension was `{ext}` instead of `.npy`)")
        else:
            print(f"Skipping {f} (not a file or directory)")


if __name__ == "__main__":
    run()
