import warnings

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from skrub import selectors as s

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")


class RegressorEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model_ridge_cv_ = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
        self.model_xgb_ = xgb.XGBRegressor(
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
        self.model_gbr_ = GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features="sqrt",
            min_samples_leaf=15,
            min_samples_split=10,
            loss="huber",
            random_state=42,
        )
        self.model_lgbm_ = LGBMRegressor(
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

    def fit(self, X, y):
        self.model_ridge_cv_.fit(X, y)
        self.model_xgb_.fit(X, y)
        self.model_gbr_.fit(X, y)
        self.model_lgbm_.fit(X, y)
        return self

    def predict(self, X):
        lgbm_preds = self.model_lgbm_.predict(X)
        gbr_preds = self.model_gbr_.predict(X)
        xgb_preds = self.model_xgb_.predict(X)
        ridge_cv_preds = self.model_ridge_cv_.predict(X)
        final_predictions = 0.3 * lgbm_preds + 0.3 * gbr_preds + 0.1 * xgb_preds + 0.3 * ridge_cv_preds
        final_predictions = final_predictions + 0.007 * final_predictions
        return final_predictions


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_error(y, y_predicted))


def sempipes_pipeline2(data_file):
    all_data_initial = pd.read_csv(data_file)
    with open(
        "experiments/house_prices_advanced_regression_techniques/data_description.txt", "r", encoding="utf-8"
    ) as f:
        data_description = f.read()

    data = skrub.var("data", all_data_initial).skb.subsample(n=100)

    data = data.drop(["Id", "Street", "PoolQC", "Utilities"], axis=1)

    y = (
        data["SalePrice"]
        .skb.mark_as_y()
        .skb.set_name("SalePrice")
        .skb.set_description("the sale price of a house to predict")
    )
    data = data.drop(["SalePrice"], axis=1).skb.mark_as_X().skb.set_description(data_description)

    data = data.with_sem_features(
        nl_prompt="""
        Create additional features that could help predict the sale price of a house.
        Consider aspects like location, size, condition, and any other relevant information.
        You way want to combine several existing features to create new ones.
        """,
        name="house_features",
        how_many=25,
    )

    data = data.skb.apply(FunctionTransformer(lambda df: df.fillna(0)), cols=(s.numeric() | s.boolean()))
    data = data.skb.apply(SimpleImputer(strategy="constant", fill_value="N/A"), cols=s.categorical())

    X = data.skb.apply(skrub.TableVectorizer())
    return X.skb.apply(RegressorEnsemble(), y=y)
