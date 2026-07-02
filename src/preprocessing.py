import pandas as pd

NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
]

ORDINAL_COLS = [
    ("BsmtExposure", {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}),
    ("BsmtFinType1", {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}),
    ("BsmtFinType2", {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}),
    ("GarageFinish", {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}),
    ("Fence",        {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}),
    ("LotShape",     {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3}),
]

NOMINAL_COLS = [
    "MSZoning", "Street", "Alley", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir",
    "Electrical", "GarageType", "MiscFeature", "SaleType", "SaleCondition",
    "MSSubClass", "Functional", "PavedDrive",
]

TEST_MODE_COLS = [
    "MSZoning", "Utilities", "Exterior1st", "Exterior2nd",
    "KitchenQual", "Functional", "SaleType",
]


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 200_000))].reset_index(drop=True)


    for col in NONE_COLS:
        train[col] = train[col].fillna("None")
        test[col]  = test[col].fillna("None")

    for col in ZERO_COLS:
        train[col] = train[col].fillna(0)
        test[col]  = test[col].fillna(0)


    train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )


    train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])
    for col in TEST_MODE_COLS:
        test[col] = test[col].fillna(train[col].mode()[0])


    train["MSSubClass"] = train["MSSubClass"].astype(str)
    test["MSSubClass"]  = test["MSSubClass"].astype(str)

    
    for col in QUAL_COLS:
        train[col] = train[col].map(QUAL_MAP)
        test[col]  = test[col].map(QUAL_MAP)

    for col, mapping in ORDINAL_COLS:
        train[col] = train[col].map(mapping)
        test[col]  = test[col].map(mapping)

    
    train = pd.get_dummies(train, columns=NOMINAL_COLS)
    test  = pd.get_dummies(test,  columns=NOMINAL_COLS)


    train, test = train.align(test, join="left", axis=1)
    test = test.fillna(0)
    test = test.drop(columns=["SalePrice"], errors="ignore")

    return train, test