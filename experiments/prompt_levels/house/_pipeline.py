import warnings
from typing import Literal

import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from skrub import selectors as s
from xgboost import XGBRegressor

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")

LearnerKind = Literal["ensemble", "fast"]

DEFAULT_HOUSE_FEATURES_PROMPT = """
        Compute a small set of simple numeric features for house sale price prediction — slightly beyond
        the bare minimum, but keep the total to about five features.

        Include total bathrooms, house age, total living area (GrLivArea plus TotalBsmtSF), and years
        since remodel. You may add one simple binary flag such as HasGarage (GarageArea > 0).

        Do not create interaction terms, quality scores, outdoor-area sums, extra amenity flags, or
        ordinal encodings of categorical columns.
        """

DEFAULT_ORDERED_FEATURES_PROMPT = """
        Replace categorical columns that have a natural quality or level ordering with separate numeric ordinal
        columns (suffix _Ordinal). Focus on quality and condition ratings (e.g. BsmtQual, KitchenQual,
        ExterQual, HeatingQC, FireplaceQu, GarageFinish, ExterCond, BsmtExposure) and property shape/slope
        attributes (e.g. LotShape, LandSlope, LandContour). Map each distinct string value to an integer
        preserving its natural order (e.g. Ex > Gd > TA > Fa > Po > None). Create one new column per encoded
        categorical — do not aggregate them into composite scores.
        """


def preprocess_house_data(data, data_description: str):
    data = data.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1, errors="ignore")
    data = data.skb.mark_as_X().skb.set_description(data_description)

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

    data = data.drop(["Heating", "RoofMatl", "Condition2", "Street", "Utilities"], axis=1, errors="ignore")

    return data.assign(
        YrSold=lambda df: df["YrSold"].where(df["YrSold"] >= df["YearBuilt"], 2009),
    )


def house_price_ensemble(X, target):
    class LGBMRegressorWithEvalMetric(LGBMRegressor):  # pylint: disable=too-many-ancestors
        def __init__(self, eval_metric=None, **kwargs):
            super().__init__(**kwargs)
            self.eval_metric = eval_metric

        def fit(self, X, y, **kwargs):  # pylint: disable=arguments-differ
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

    return xgb_predictions * 0.45 + lgbm_predictions * 0.55


def house_price_fast_learner(X, target):
    """Lightweight regressor for COLOPRO search (similar cost to churn/fraudbaskets)."""
    model = HistGradientBoostingRegressor(max_depth=4, max_iter=100, random_state=0)
    return X.skb.apply(model, y=target)


def apply_house_price_learner(X, target, learner: LearnerKind = "ensemble"):
    if learner == "fast":
        return house_price_fast_learner(X, target)
    if learner == "ensemble":
        return house_price_ensemble(X, target)
    raise ValueError(f"Unknown learner: {learner}. Choose from 'ensemble' or 'fast'.")


def sempipes_pipeline(house_features_prompt: str | None = None, learner: LearnerKind = "ensemble"):
    house_features_prompt = house_features_prompt or DEFAULT_HOUSE_FEATURES_PROMPT
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
    data = data.drop(["SalePrice"], axis=1)
    data = preprocess_house_data(data, data_description)

    data = data.sem_gen_features(
        nl_prompt=house_features_prompt,
        name="house_features",
        how_many=10,
    )

    data = data.sem_gen_features(
        nl_prompt=DEFAULT_ORDERED_FEATURES_PROMPT,
        name="ordered_features",
        how_many=10,
    )

    vectorizer = skrub.TableVectorizer(
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        high_cardinality=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )

    X = data.skb.apply(vectorizer)
    return apply_house_price_learner(X, target, learner=learner)
