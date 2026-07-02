"""Вспомогательные функции общего назначения."""

import numpy as np
from sklearn.metrics import make_scorer


def rmse_log(y_true, y_pred) -> float:
    """RMSE между log1p(true) и log1p(pred) — ровно метрика соревнования."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


rmse_log_scorer = make_scorer(rmse_log, greater_is_better=False)
