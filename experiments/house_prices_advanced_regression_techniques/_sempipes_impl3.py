import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_log_error(y, y_predicted))


def sempipes_pipeline3(data_file):
    all_data_initial = pd.read_csv(data_file)
    with open(
        "experiments/house_prices_advanced_regression_techniques/data_description.txt", "r", encoding="utf-8"
    ) as f:
        data_description = f.read()

    data = skrub.var("data", all_data_initial).skb.subsample(n=100)

    y = (
        data["SalePrice"]
        .skb.mark_as_y()
        .skb.set_name("SalePrice")
        .skb.set_description("the sale price of a house to predict")
    )
    data = data.drop(["SalePrice"], axis=1).skb.mark_as_X().skb.set_description(data_description)

    data = data.sem_gen_features(
        nl_prompt="""
        Create additional features that could help predict the sale price of a house.
        Consider aspects like location, size, condition, and any other relevant information.
        You way want to combine several existing features to create new ones.
        """,
        name="house_features",
        how_many=25,
    )

    data = data.drop(["Id", "PoolQC", "PoolArea"], axis=1)

    # drop features with little information based on visualizations
    data = data.drop(
        [
            "BsmtFinSF2",
            "LowQualFinSF",
            "BsmtHalfBath",
            "KitchenAbvGr",
            "EnclosedPorch",
            "3SsnPorch",
            "MiscVal",
            "Street",
            "Utilities",
            "Condition2",
            "RoofMatl",
            "Heating",
            "MiscFeature",
        ],
        axis=1,
    )

    data = data.drop(["MSSubClass", "OverallCond", "ScreenPorch", "MoSold", "YrSold"], axis=1)

    X = data.skb.apply(skrub.TableVectorizer())

    xgb = XGBRegressor(random_state=42)

    parameter_grid = {
        "n_estimators": [500, 525, 550],
        "max_depth": [3],
        "learning_rate": [0.04, 0.05],
        "colsample_bytree": [0.25, 0.3],
        "subsample": [0.65, 0.7, 0.75],
    }

    xgb_grid = GridSearchCV(estimator=xgb, param_grid=parameter_grid, cv=3, verbose=False, n_jobs=-1)

    return X.skb.apply(xgb_grid, y=y)
