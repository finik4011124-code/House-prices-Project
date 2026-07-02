import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from catboost import CatBoostRegressor


def _load_params() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["model"]


def train_final_model(X: pd.DataFrame, y: pd.Series) -> CatBoostRegressor:
    params = _load_params()
    model = CatBoostRegressor(**params)
    model.fit(X, np.log1p(y))
    return model


def predict(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(X))