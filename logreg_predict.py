import pandas as pd  # type: ignore
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset for training")

    args = parser.parse_args()

    try:
        test = pd.read_csv(args.dataset).fillna(0)
    except Exception as e:
        print("Something wrong with dataset file:", e.args)
        exit(1)

    try:
        with open("weights", "rb") as f:
            model = pickle.load(f)
            LR = model["model"]
            X_mean = model["X_mean"]
            X_std = model["X_std"]
    except Exception as e:
        print("Something wrong with weights file:", e.args)
        exit(1)

    try:
        if len(X_mean) == 6:
            X = test[
                [
                    "Charms",
                    "Flying",
                    "Divination",
                    "Ancient Runes",
                    "Astronomy",
                    "Herbology",
                ]
            ].values
        else:
            X = test[test.columns[6:]].values
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    except Exception as e:
        print("Something wrong with test:", e.args)
        exit(1)

    y = LR.predict(X)
    test["Hogwarts House"] = y
    ans = test[["Index", "Hogwarts House"]]
    ans.to_csv("houses.csv", index=False)
