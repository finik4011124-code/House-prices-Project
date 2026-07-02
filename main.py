from pathlib import Path
import yaml
import pandas as pd

from src.data import load_train, load_test
from src.preprocessing import preprocess
from src.features import engineer_features
from src.model import train_final_model, predict

ROOT = Path(__file__).parent

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = ROOT / "data"
OUTPUT   = ROOT / cfg["data"]["output_path"]


def main() -> None:
    train_raw = load_train(DATA_DIR)
    test_raw  = load_test(DATA_DIR)

    train, test = preprocess(train_raw, test_raw)
    train, test = engineer_features(train, test)

    X = train.drop(columns=["Id", "SalePrice"])
    y = train["SalePrice"]
    ids = test["Id"]
    X_test = test.drop(columns=["Id"])

    model = train_final_model(X, y)
    preds = predict(model, X_test)

    pd.DataFrame({"Id": ids, "SalePrice": preds}).to_csv(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()