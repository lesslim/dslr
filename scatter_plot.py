import bears as br
import matplotlib.pyplot as plt  # type: ignore
import argparse


def scatter_plot(df):
    plt.figure(figsize=(12, 9))
    plt.title("How features correlate")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.scatter(
        df[df.columns[0]], df[df.columns[1]], color="blue", label="students", s=10
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        df = br.read_csv("datasets/dataset_train.csv")
    except Exception as e:
        print("Something wrong with train file:", e.args)
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--one",
        "-o",
        default=False,
        action="store_true",
        help="Display right correlation.",
    )
    parser.add_argument(
        "--all",
        "-a",
        default=False,
        action="store_true",
        help="Display all correlations.",
    )

    flags = parser.parse_args()

    try:
        if flags.one or not flags.all:
            right_df = df[["Astronomy", "Defense Against the Dark Arts"]].dropna()
            scatter_plot(right_df)
        if flags.all:
            for i in range(6, 18):
                for j in range(i + 1, 19):
                    iter_df = df[[df.columns[i], df.columns[j]]].dropna()
                    scatter_plot(iter_df)
    except Exception as e:
        print("Something wrong with train:", e.args)
        exit(1)
