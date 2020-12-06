import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import argparse
from logreg import LogisticRegression


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset for training")
    parser.add_argument("--lr", "-l", type=float, help="learning rate", default=0.01)
    parser.add_argument(
        "--all",
        "-a",
        default=False,
        action="store_true",
        help="use all features for training",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, help="number of epochs", default=100
    )

    args = parser.parse_args()

    if args.lr >= 1 or args.lr <= 0:
        print("lr should be between 0 and 1.")
        exit(1)

    if args.epochs < 1:
        print("Number of epochs should be > 0.")
        exit(1)

    try:
        train = pd.read_csv(args.dataset).fillna(0)
    except Exception as e:
        print("Something wrong with dataset file:", e.args)
        exit(1)

    try:
        if args.all:
            features = train.columns[6:]
        else:
            features = [
                "Charms",
                "Flying",
                "Divination",
                "Ancient Runes",
                "Astronomy",
                "Herbology",
            ]

        X = train[features].values
        y = train["Hogwarts House"]

        np.nanmean(X, axis=0)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X = (X - X_mean) / X_std
    except Exception as e:
        print("Something wrong with train:", e.args)
        exit(1)

    LR = LogisticRegression(args.lr, args.epochs)
    LR.fit(X, y)

    acc, err = LR.accuracy(y, LR.predict(X))
    print("Accuracy on train:", acc)
    print("Number of errors: ", err)
    LR.save_model(X_mean, X_std)
