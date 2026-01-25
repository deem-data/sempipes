# https://www.kaggle.com/code/redaabdou/house-prices-solution-data-cleaning-ml
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

all_data_initial = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_log_error(y, y_predicted))


scores = []
seed = 42
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_index, (train_idx, test_idx) in enumerate(kf.split(all_data_initial)):
    train = all_data_initial.iloc[train_idx]
    test = all_data_initial.iloc[test_idx]
    np.random.seed(seed)
    random.seed(seed)

    #`train, test = train_test_split(all_data_initial, test_size=0.5, random_state=seed)

    data = pd.concat([train.iloc[:, :-1], test], axis=0)
    data = data.drop(columns=["Id", "Street", "PoolQC", "Utilities"], axis=1)

    train_lot_median = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.median())
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(train_lot_median[x.name] if x.name in train_lot_median.index else 0)
    )
    train_msz_mode = train.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else "Other"
    )
    data["MSZoning"] = data.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.fillna(
            train_msz_mode[x.name]
            if x.name in train_msz_mode.index
            else (x.mode()[0] if len(x.mode()) > 0 else "Other")
        )
    )

    features = [
        "Electrical",
        "KitchenQual",
        "SaleType",
        "Exterior2nd",
        "Exterior1st",
        "Alley",
        "Fence",
        "MiscFeature",
        "FireplaceQu",
        "GarageCond",
        "GarageQual",
        "GarageFinish",
        "GarageType",
        "BsmtCond",
        "BsmtExposure",
        "BsmtQual",
        "BsmtFinType2",
        "BsmtFinType1",
        "MasVnrType",
    ]
    for name in features:
        data[name].fillna("Other", inplace=True)

    data["Functional"] = data["Functional"].fillna("typ")

    zero = [
        "GarageArea",
        "GarageYrBlt",
        "MasVnrArea",
        "BsmtHalfBath",
        "BsmtHalfBath",
        "BsmtFullBath",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "GarageCars",
    ]
    for name in zero:
        data[name].fillna(0, inplace=True)

    data.loc[data["MSSubClass"] == 60, "MSSubClass"] = 0
    data.loc[(data["MSSubClass"] == 20) | (data["MSSubClass"] == 120), "MSSubClass"] = 1
    data.loc[data["MSSubClass"] == 75, "MSSubClass"] = 2
    data.loc[(data["MSSubClass"] == 40) | (data["MSSubClass"] == 70) | (data["MSSubClass"] == 80), "MSSubClass"] = 3
    data.loc[
        (data["MSSubClass"] == 50)
        | (data["MSSubClass"] == 85)
        | (data["MSSubClass"] == 90)
        | (data["MSSubClass"] == 160)
        | (data["MSSubClass"] == 190),
        "MSSubClass",
    ] = 4
    data.loc[(data["MSSubClass"] == 30) | (data["MSSubClass"] == 45) | (data["MSSubClass"] == 180), "MSSubClass"] = 5
    data.loc[(data["MSSubClass"] == 150), "MSSubClass"] = 6

    object_features = data.select_dtypes(include="object").columns

    def dummies(d):
        dummies_df = pd.DataFrame()
        object_features = d.select_dtypes(include="object").columns
        for name in object_features:
            dummies = pd.get_dummies(d[name], drop_first=False)
            dummies = dummies.add_prefix("{}_".format(name))
            dummies_df = pd.concat([dummies_df, dummies], axis=1)
        return dummies_df

    dummies_data = dummies(data)
    data = data.drop(columns=object_features, axis=1)

    final_data = pd.concat([data, dummies_data], axis=1)

    train_data = final_data.iloc[: train.shape[0], :]
    test_data = final_data.iloc[train.shape[0] :, :]

    X = train_data
    X.drop(columns=["SalePrice"], inplace=True)
    y = train.loc[:, "SalePrice"]

    X_test = test_data
    y_test = test_data["SalePrice"]
    X_test.drop(columns=["SalePrice"], inplace=True)

    train_data.fillna(0, inplace=True)

    from sklearn.linear_model import ElasticNet, LassoCV, Ridge, RidgeCV

    model_las_cv = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
    model_las_cv.fit(X, y)
    las_cv_preds = model_las_cv.predict(test_data)

    model_ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
    model_ridge_cv.fit(X, y)
    ridge_cv_preds = model_ridge_cv.predict(X_test)

    model_ridge = Ridge(alpha=10, solver="auto")
    model_ridge.fit(X, y)
    ridge_preds = model_ridge.predict(test_data)

    model_en = ElasticNet(random_state=1, alpha=0.00065, max_iter=3000)
    model_en.fit(X, y)
    en_preds = model_en.predict(test_data)

    import xgboost as xgb

    model_xgb = xgb.XGBRegressor(
        learning_rate=0.01,
        n_estimators=3460,
        max_depth=3,
        min_child_weight=0,
        gamma=0,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="reg:linear",
        nthread=-1,
        scale_pos_weight=1,
        seed=27,
        reg_alpha=0.00006,
    )
    model_xgb.fit(X, y)
    xgb_preds = model_xgb.predict(X_test)

    from sklearn.ensemble import GradientBoostingRegressor

    model_gbr = GradientBoostingRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=15,
        min_samples_split=10,
        loss="huber",
        random_state=42,
    )
    model_gbr.fit(X, y)
    gbr_preds = model_gbr.predict(X_test)

    from lightgbm import LGBMRegressor

    model_lgbm = LGBMRegressor(
        objective="regression",
        num_leaves=4,
        learning_rate=0.01,
        n_estimators=5000,
        max_bin=200,
        bagging_fraction=0.75,
        bagging_freq=5,
        bagging_seed=7,
        feature_fraction=0.2,
        feature_fraction_seed=7,
        verbose=-1,
    )
    model_lgbm.fit(X, y)
    lgbm_preds = model_lgbm.predict(X_test)

    final_predictions = 0.3 * lgbm_preds + 0.3 * gbr_preds + 0.1 * xgb_preds + 0.3 * ridge_cv_preds
    final_predictions = final_predictions + 0.007 * final_predictions
    score = rmsle(y_test, final_predictions)
    print(f"RMSLE on {fold_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
