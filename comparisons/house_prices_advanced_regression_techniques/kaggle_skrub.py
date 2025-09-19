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

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_error(y, y_predicted))


all_data_initial = pd.read_csv("comparisons/house_prices_advanced_regression_techniques/data.csv")

data = skrub.var("data", all_data_initial).skb.subsample(n=100)

data = data.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

target = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()
data = data.drop(["SalePrice"], axis=1).skb.mark_as_X()

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
# object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])

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
    # now compute derived columns using the updated YrSold
    Age_House=lambda df: df["YrSold"] - df["YearBuilt"],
    TotalBsmtBath=lambda df: df["BsmtFullBath"] * 1.5,
    TotalBath=lambda df: df["FullBath"] + df["HalfBath"] * 0.5,
    TotalSA=lambda df: df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"],
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
PavedDrive = {"N": 0, "P": 1, "Y": 2}

data = data.assign(
    ExterQual=lambda df: df["ExterQual"].map(bin_map),
    ExterCond=lambda df: df["ExterCond"].map(bin_map),
    BsmtCond=lambda df: df["BsmtCond"].map(bin_map),
    BsmtQual=lambda df: df["BsmtQual"].map(bin_map),
    HeatingQC=lambda df: df["HeatingQC"].map(bin_map),
    KitchenQual=lambda df: df["KitchenQual"].map(bin_map),
    FireplaceQu=lambda df: df["FireplaceQu"].map(bin_map),
    GarageQual=lambda df: df["GarageQual"].map(bin_map),
    GarageCond=lambda df: df["GarageCond"].map(bin_map),
    CentralAir=lambda df: df["CentralAir"].map(bin_map),
    LotShape=lambda df: df["LotShape"].map(bin_map),
    BsmtExposure=lambda df: df["BsmtExposure"].map(bin_map),
    BsmtFinType1=lambda df: df["BsmtFinType1"].map(bin_map),
    BsmtFinType2=lambda df: df["BsmtFinType2"].map(bin_map),
    PavedDrive=lambda df: df["PavedDrive"].map(PavedDrive),
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

    def fit(self, X, y, **kwargs):  # pylint: disable=redefined-outer-name, arguments-differ
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

predictions = xgb_predictions * 0.45 + lgbm_predictions * 0.55

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = predictions.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])
    score = rmsle(split["test"]["_skrub_y"], y_pred)
    print(f"RMSLE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
