import numpy as np
from sklearn.base import BaseEstimator


def predict(model: BaseEstimator, features: list[list[float]]) -> list[float]:
    """Run model inference on a feature matrix and return predictions as Python floats."""
    X = np.array(features)
    return model.predict(X).tolist()
