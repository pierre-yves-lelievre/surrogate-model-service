import numpy as np
from sklearn.base import BaseEstimator


def predict(model: BaseEstimator, features: list[list[float]]) -> list[float]:
    X = np.array(features)
    return model.predict(X).tolist()
