import pandas as pd


def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in [train, test]:
        df["TotalSF"]      = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["HouseAge"]     = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"]     = df["YrSold"] - df["YearRemodAdd"]
        df["IsRemodeled"]  = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
        df["TotalBath"]    = (df["FullBath"] + df["BsmtFullBath"]
                              + 0.5 * df["HalfBath"] + 0.5 * df["BsmtHalfBath"])
        df["TotalPorch"]   = (df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"]
                              + df["3SsnPorch"] + df["ScreenPorch"])
        df["HasPool"]      = (df["PoolArea"]    > 0).astype(int)
        df["HasGarage"]    = (df["GarageArea"]  > 0).astype(int)
        df["HasFireplace"] = (df["Fireplaces"]  > 0).astype(int)
        df["HasBasement"]  = (df["TotalBsmtSF"] > 0).astype(int)

    return train, test