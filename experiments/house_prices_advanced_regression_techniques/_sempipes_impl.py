import warnings

import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from skrub import selectors as s
from xgboost import XGBRegressor

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_error(y, y_predicted))


def sempipes_pipeline():
    with open(
        "experiments/house_prices_advanced_regression_techniques/data_description.txt", "r", encoding="utf-8"
    ) as f:
        data_description = f.read()

    data = skrub.var("data")

    data = data.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

    target = (
        data["SalePrice"]
        .skb.apply_func(np.log1p)
        .skb.mark_as_y()
        .skb.set_name("SalePrice")
        .skb.set_description("the sale price of a house to predict")
    )
    data = data.drop(["SalePrice"], axis=1).skb.mark_as_X().skb.set_description(data_description)

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
    fill_with_none = FunctionTransformer(lambda df: df.fillna("None"))

    data = data.skb.apply(fill_with_none, cols=s.cols(*columns_None))

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

    mode_imputer = SimpleImputer(strategy="most_frequent")
    data = data.skb.apply(mode_imputer, cols=s.cols(*columns_with_lowNA))

    data = data.assign(
        GarageYrBlt=data["GarageYrBlt"].fillna(data["YrSold"] - 35), LotFrontage=data["LotFrontage"].fillna(68)
    )

    fill_with_zero = FunctionTransformer(lambda df: df.fillna(0))
    data = data.skb.apply(fill_with_zero, cols=s.numeric())

    data = data.drop(["Heating", "RoofMatl", "Condition2", "Street", "Utilities"], axis=1)

    data = data.assign(
        # replace bad YrSold values with 2009 when YrSold < YearBuilt
        YrSold=lambda df: df["YrSold"].where(df["YrSold"] >= df["YearBuilt"], 2009),
    )

    data = data.sem_gen_features(
        nl_prompt="""
        Compute additional features from the house attributes, e.g. counting the total number of bath rooms, bed rooms,
        computing the age of the house and the overall square of the house etc. Also consider meaningful combinations of 
        the above.
        """,
        name="house_features",
        how_many=10,
    )

    data = data.sem_gen_features(
        nl_prompt="""
        Replace the categorical features (that have string values) with an ordinal variant (e.g., with an integer range of
        numbers) if there is a meaningful order in the values. Do this so that it becomes easier for tree-based models
        to find good splits on these attributes.
        """,
        name="ordered_features",
        how_many=10,
    )

    vectorizer = skrub.TableVectorizer(
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        high_cardinality=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )

    X = data.skb.apply(vectorizer)

    class LGBMRegressorWithEvalMetric(LGBMRegressor):  # pylint: disable=too-many-ancestors
        def __init__(self, eval_metric=None, **kwargs):
            super().__init__(**kwargs)
            self.eval_metric = eval_metric

        def fit(self, X, y, **kwargs):  # pylint: disable=arguments-differ
            # inject eval_metric unless explicitly overridden
            if self.eval_metric is not None and "eval_metric" not in kwargs:
                kwargs["eval_metric"] = self.eval_metric
            return super().fit(X, y, **kwargs)

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

    lgbm = LGBMRegressorWithEvalMetric(
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
        eval_metric="rmse",
    )

    lgbm_predictions = X.skb.apply(lgbm, y=target)
    xgb_predictions = X.skb.apply(xgb, y=target)

    mode = skrub.eval_mode()
    lgbm_predictions = lgbm_predictions.skb.apply_func(lambda pred, m: 0 if m == "fit" else pred, m=mode)
    xgb_predictions = xgb_predictions.skb.apply_func(lambda pred, m: 0 if m == "fit" else pred, m=mode)

    predictions = xgb_predictions * 0.45 + lgbm_predictions * 0.55
    return predictions
