import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from skrub import DataOp, TableVectorizer

from experiments.optimize_multiple import TestPipeline

warnings.filterwarnings("ignore")

_OP_MEANINGFUL1 = "op_house_features"
_OP_MEANINGFUL2 = "op_categorical_features"
_OP_MEANINGLESS = "op_noise_features"

_DATA_PATH = Path("experiments/house_prices_advanced_regression_techniques/data.csv")
_DESCRIPTION_PATH = Path("experiments/house_prices_advanced_regression_techniques/data_description.txt")


def rmsle(y_true, y_predicted):
    return np.sqrt(mean_squared_error(y_true, y_predicted))


class HousePricePipeline(TestPipeline):
    OPERATOR_NAMES = [_OP_MEANINGFUL1, _OP_MEANINGFUL2, _OP_MEANINGLESS]

    @property
    def name(self) -> str:
        return "houseprices"

    def score(self, y_true, y_pred) -> float:
        # Higher is better for TestPipeline (negated RMSLE on log1p prices).
        return -rmsle(y_true, y_pred)

    @property
    def scoring(self) -> str:
        return "r2"

    def pipeline_with_all_data(self, seed) -> tuple[DataOp, dict]:
        del seed
        dataset = pd.read_csv(_DATA_PATH)
        features, labels = _split_features_and_labels(dataset)
        additional_env_variables = {
            "data": features,
            "labels": labels,
        }
        return _pipeline(), additional_env_variables

    def pipeline_with_train_data(self, seed) -> tuple[DataOp, dict]:
        dataset = pd.read_csv(_DATA_PATH)
        train_data, _ = train_test_split(dataset, test_size=TestPipeline.TEST_SIZE, random_state=seed)
        features, labels = _split_features_and_labels(train_data)
        additional_env_variables = {
            "data": features,
            "labels": labels,
        }
        return _pipeline(), additional_env_variables


def _split_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    labels = np.log1p(df["SalePrice"])
    features = df.drop(columns=["SalePrice"])
    if "Id" in features:
        features = features.drop(columns=["Id"])
    return features, labels


def _minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string", "category"]).columns:
        out[col] = out[col].fillna("None")
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].fillna(0)
    return out


def _pipeline() -> DataOp:
    with open(_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
        data_description = f.read()

    data = skrub.var("data")
    labels = skrub.var("labels")

    labels = labels.skb.mark_as_y().skb.set_description("Log scaled sale price of a house to predict")
    data = data.skb.mark_as_X().skb.set_description(data_description)

    # Minimal cleaning so semantic operators have headroom to improve CV score.
    data = data.skb.apply(FunctionTransformer(_minimal_clean))
    cleaned = data
    cleaned_copy = cleaned.copy(deep=True)

    # Parallel branches — each operator credited independently; noise sees only random seeds.
    noise_features = cleaned_copy.assign(
        random_col1=lambda df: np.random.rand(len(df)),
        random_col2=lambda df: np.random.choice(["A", "B", "C"], size=len(df)),
    )[["random_col1", "random_col2"]]

    noise_features = noise_features.sem_gen_features(
        nl_prompt=(
            "Generate at numeric or categorical columns using ONLY numpy random permutations "
            "only from random_col1 and random_col2 to destroy the data quality. "
            "Do not read, merge, or derive from any other dataframe columns. "
            "Features must be pure noise and destroy predictive signal for house prices."
        ),
        name=_OP_MEANINGLESS,
        how_many=2,
    )

    house_features = cleaned.sem_gen_features(
        nl_prompt="""
        Compute strong predictive features from the house attributes: total bathrooms, total bedrooms,
        house age (YrSold - YearBuilt), total living area, garage capacity, basement area,
        and ratios such as living area per room. Prioritize features that correlate with sale price.
        """,
        name=_OP_MEANINGFUL1,
        how_many=10,
    )

    categorical_features = cleaned.sem_gen_features(
        nl_prompt="""
        Replace categorical string columns with ordinal encodings where values have a natural order
        (e.g. quality ratings, exposure levels). This should make it easier for linear models to split
        on quality and condition attributes that drive sale price.
        """,
        name=_OP_MEANINGFUL2,
        how_many=10,
    )

    data = house_features.skb.concat([categorical_features, noise_features], axis=1)

    vectorizer = TableVectorizer(
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        high_cardinality=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )
    data = data.skb.apply(vectorizer)

    return data.skb.apply(LinearRegression(), y=labels)