from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """Compute MSE, RMSE, MAE, and R² for a set of predictions against ground-truth targets."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
