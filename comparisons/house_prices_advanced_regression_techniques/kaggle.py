# https://www.kaggle.com/code/jesucristo/1-house-prices-solution-top-1
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_error(y, y_predicted))


all_data_initial = pd.read_csv("comparisons/house_prices_advanced_regression_techniques/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    df = all_data_initial.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

    object_columns_df = df.select_dtypes(include=["object"])
    numerical_columns_df = df.select_dtypes(exclude=["object"])

    columns_None = [
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "FireplaceQu",
        "GarageCond",
    ]
    object_columns_df[columns_None] = object_columns_df[columns_None].fillna("None")

    columns_with_lowNA = [
        "MSZoning",
        "Utilities",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "Electrical",
        "KitchenQual",
        "Functional",
        "SaleType",
    ]
    # fill missing values for each column (using its own most frequent value)
    object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(
        object_columns_df.mode().iloc[0]
    )

    numerical_columns_df["GarageYrBlt"] = numerical_columns_df["GarageYrBlt"].fillna(
        numerical_columns_df["YrSold"] - 35
    )
    numerical_columns_df["LotFrontage"] = numerical_columns_df["LotFrontage"].fillna(68)

    numerical_columns_df = numerical_columns_df.fillna(0)
    object_columns_df = object_columns_df.drop(["Heating", "RoofMatl", "Condition2", "Street", "Utilities"], axis=1)
    numerical_columns_df["Age_House"] = numerical_columns_df["YrSold"] - numerical_columns_df["YearBuilt"]

    numerical_columns_df.loc[numerical_columns_df["YrSold"] < numerical_columns_df["YearBuilt"], "YrSold"] = 2009
    numerical_columns_df["Age_House"] = numerical_columns_df["YrSold"] - numerical_columns_df["YearBuilt"]

    numerical_columns_df["TotalBsmtBath"] = (
        numerical_columns_df["BsmtFullBath"] + numerical_columns_df["BsmtFullBath"] * 0.5
    )
    numerical_columns_df["TotalBath"] = numerical_columns_df["FullBath"] + numerical_columns_df["HalfBath"] * 0.5
    numerical_columns_df["TotalSA"] = (
        numerical_columns_df["TotalBsmtSF"] + numerical_columns_df["1stFlrSF"] + numerical_columns_df["2ndFlrSF"]
    )

    bin_map = {
        "TA": 2,
        "Gd": 3,
        "Fa": 1,
        "Ex": 4,
        "Po": 1,
        "Y": 1,
        "N": 0,
        "Reg": 3,
        "IR1": 2,
        "IR2": 1,
        "IR3": 0,
        "None": 0,
        "No": 2,
        "Mn": 2,
        "Av": 3,
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6,
    }
    object_columns_df["ExterQual"] = object_columns_df["ExterQual"].map(bin_map)
    object_columns_df["ExterCond"] = object_columns_df["ExterCond"].map(bin_map)
    object_columns_df["BsmtCond"] = object_columns_df["BsmtCond"].map(bin_map)
    object_columns_df["BsmtQual"] = object_columns_df["BsmtQual"].map(bin_map)
    object_columns_df["HeatingQC"] = object_columns_df["HeatingQC"].map(bin_map)
    object_columns_df["KitchenQual"] = object_columns_df["KitchenQual"].map(bin_map)
    object_columns_df["FireplaceQu"] = object_columns_df["FireplaceQu"].map(bin_map)
    object_columns_df["GarageQual"] = object_columns_df["GarageQual"].map(bin_map)
    object_columns_df["GarageCond"] = object_columns_df["GarageCond"].map(bin_map)
    object_columns_df["CentralAir"] = object_columns_df["CentralAir"].map(bin_map)
    object_columns_df["LotShape"] = object_columns_df["LotShape"].map(bin_map)
    object_columns_df["BsmtExposure"] = object_columns_df["BsmtExposure"].map(bin_map)
    object_columns_df["BsmtFinType1"] = object_columns_df["BsmtFinType1"].map(bin_map)
    object_columns_df["BsmtFinType2"] = object_columns_df["BsmtFinType2"].map(bin_map)

    PavedDrive = {"N": 0, "P": 1, "Y": 2}
    object_columns_df["PavedDrive"] = object_columns_df["PavedDrive"].map(PavedDrive)

    # Select categorical features
    rest_object_columns = object_columns_df.select_dtypes(include=["object"])
    # Using One hot encoder
    object_columns_df = pd.get_dummies(object_columns_df, columns=rest_object_columns.columns)

    df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1, sort=False)

    df_final = df_final.drop(
        [
            "Id",
        ],
        axis=1,
    )

    df_train, df_test = train_test_split(df_final, test_size=0.5, random_state=seed)

    y_train = np.log1p(df_train["SalePrice"])
    x_train = df_train.drop(["SalePrice"], axis=1)

    y_test = np.log1p(df_test["SalePrice"])
    x_test = df_test.drop(["SalePrice"], axis=1)

    xgb = XGBRegressor(
        booster="gbtree",
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=0.6,
        gamma=0,
        importance_type="gain",
        learning_rate=0.01,
        max_delta_step=0,
        max_depth=4,
        min_child_weight=1.5,
        n_estimators=2400,
        n_jobs=1,
        nthread=None,
        objective="reg:linear",
        reg_alpha=0.6,
        reg_lambda=0.6,
        scale_pos_weight=1,
        silent=None,
        subsample=0.8,
        verbosity=1,
    )

    lgbm = LGBMRegressor(
        objective="regression",
        num_leaves=4,
        learning_rate=0.01,
        n_estimators=12000,
        max_bin=200,
        bagging_fraction=0.75,
        bagging_freq=5,
        bagging_seed=7,
        feature_fraction=0.4,
        verbose=-1,
    )

    xgb.fit(x_train, y_train)
    lgbm.fit(x_train, y_train, eval_metric="rmse")

    lgmb_predictions = lgbm.predict(x_test)
    xgb_predictions = xgb.predict(x_test)
    y_pred = xgb_predictions * 0.45 + lgmb_predictions * 0.55

    score = rmsle(y_test, y_pred)
    print(f"RMSLE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
