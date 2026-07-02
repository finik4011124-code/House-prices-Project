from pathlib import Path
import pandas as pd


def load_train(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "train.csv")


def load_test(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "test.csv")