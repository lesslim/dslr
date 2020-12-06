import bears as br
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import argparse


def pair_plot(df):
    y = df["Hogwarts House"].map(
        {
            "Ravenclaw": "blue",
            "Slytherin": "green",
            "Gryffindor": "red",
            "Hufflepuff": "yellow",
        }
    )
    axes = pd.plotting.scatter_matrix(df, c=y,
                                      figsize=(15, 12), s=10, diagonal="kde")
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(45)
        ax.yaxis.label.set_ha("right")
        ax.set_xticklabels([], fontdict=None, minor=False)
        ax.set_yticklabels([], fontdict=None, minor=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--one",
        "-o",
        default=False,
        action="store_true",
        help="Display right correlation.",
    )

    args = parser.parse_args()

    try:
        df = br.read_csv("datasets/dataset_train.csv")
        if args.one:
            pair_plot(df[
                [
                    "Hogwarts House",
                    "Charms",
                    "Flying",
                    "Divination",
                    "Ancient Runes",
                    "Astronomy",
                    "Herbology",
                ]
            ].dropna())
        else:
            pair_plot(df[["Hogwarts House", *df.columns[6:]]].dropna())
    except Exception as e:
        print("Something wrong with train:", ";".join(e.args))
        exit(1)
