import numpy as np  # type: ignore
import pickle
from typing import Tuple


class LogisticRegression:
    def __init__(self, lr: float = 0.01, epochs: int = 21):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        m = X.shape[1]
        self.targets = np.unique(y)

        self.weights = np.zeros(len(self.targets) * m) \
            .reshape(len(self.targets), m)

        y_hot = np.zeros((len(self.targets), len(y)))
        for i in range(len(self.targets)):
            y_hot[i] = np.where(y == self.targets[i], 1, 0)

        for _ in range(self.epochs):
            pred = self.sigmoid(self.weights.dot(X.T))
            self.weights -= (self.lr / m) * (pred - y_hot).dot(X)

    @staticmethod
    def sigmoid(x: float):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def accuracy(y, y_pred) -> Tuple[float, int]:
        same = (y.values == y_pred).sum()
        return same / len(y), len(y) - same

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        pred = self.sigmoid(self.weights.dot(X.T)).T
        return [self.targets[x] for x in pred.argmax(1)]

    def save_model(self, X_mean, X_std):
        data = {"model": self, "X_mean": X_mean, "X_std": X_std}
        with open("weights", "wb") as f:
            pickle.dump(data, f)
