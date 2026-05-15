import numpy as np
from sklearn.ensemble import RandomForestRegressor

from app.logging_setup import get_logger

log = get_logger(__name__)


def train_model(features: list[list[float]], targets: list[float]) -> RandomForestRegressor:
    """Fit a RandomForestRegressor on the provided dataset and return it."""
    X = np.array(features)
    y = np.array(targets)

    log.info("training_started", n_samples=X.shape[0], n_features=X.shape[1])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    log.info("training_completed", n_samples=X.shape[0], n_features=X.shape[1])
    return model
