import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
import argparse


def plot_embedding(X, df, title=None):
    plt.figure(figsize=(12, 9))
    colors = ["blue", "green", "brown", "olive",
              "cyan", "lime", "red", "yellow"]
    labels = [
        "true Ravenclaw",
        "true Slytherin",
        "true Gryffindor",
        "true Hufflepuff",
        "Ravenclaw",
        "Slytherin",
        "Gryffindor",
        "Hufflepuff",
    ]
    for i in range(8):
        idx = np.where(df["Hogwarts House"] == i)
        plt.scatter(X[idx, 0], X[idx, 1], color=colors[i], label=labels[i])
    plt.legend(loc="lower right", fontsize=10)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title, fontsize=15)
    plt.show()


def t_SNE(X):
    tsne = TSNE(n_components=2, init="pca", n_iter=500)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, X, "Visualization of houses (true and predicted).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        "-a",
        default=False,
        action="store_true",
        help="Display all correlations.",
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv("datasets/dataset_train.csv",
                         index_col="Index").dropna()
        df["Hogwarts House"] = df["Hogwarts House"].map(
            {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
        )
        if args.all:
            train = df[["Hogwarts House", *df.columns[5:]]].dropna()
        else:
            train = df[
                [
                    "Hogwarts House",
                    "Charms",
                    "Flying",
                    "Divination",
                    "Ancient Runes",
                    "Astronomy",
                    "Herbology",
                ]
            ].dropna()
    except Exception as e:
        print("Something wrong with train:", e.args)
        exit(1)

    try:
        test = pd.read_csv("datasets/dataset_test.csv", index_col="Index")
        y = pd.read_csv("houses.csv", index_col="Index")
        test["Hogwarts House"] = y["Hogwarts House"].map(
            {"Ravenclaw": 4, "Slytherin": 5, "Gryffindor": 6, "Hufflepuff": 7}
        )
        if args.all:
            test = test[["Hogwarts House", *test.columns[5:]]].dropna()
        else:
            test = test[
                [
                    "Hogwarts House",
                    "Charms",
                    "Flying",
                    "Divination",
                    "Ancient Runes",
                    "Astronomy",
                    "Herbology",
                ]
            ].dropna()
    except Exception as e:
        print("Something wrong with test or houses:", e.args)
        exit(1)

    X = train.append(test, ignore_index=True)

    t_SNE(X)
