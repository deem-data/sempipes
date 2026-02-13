import warnings

import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

from experiments.colopro import TestPipeline

warnings.filterwarnings("ignore")


class FraudBasketsPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "fraudbaskets"

    def score(self, y_true, y_pred) -> float:
        return f1_score(y_true, y_pred)

    @property
    def scoring(self) -> str:
        return "f1"

    def pipeline_with_all_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_credit_fraud()

        all_baskets = dataset["baskets"]
        nonfraudulent_baskets = all_baskets[all_baskets.fraud_flag == 0]
        fraudulent_baskets = all_baskets[all_baskets.fraud_flag == 1]

        baskets = pd.concat([nonfraudulent_baskets.iloc[:4000], fraudulent_baskets.iloc[:1000]])

        additional_env_variables = {
            "data": baskets[["ID"]],
            "labels": baskets["fraud_flag"],
            "products": dataset["products"],
        }

        return _pipeline(), additional_env_variables

    def pipeline_with_train_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_credit_fraud()

        all_baskets = dataset["baskets"]
        nonfraudulent_baskets = all_baskets[all_baskets.fraud_flag == 0]
        fraudulent_baskets = all_baskets[all_baskets.fraud_flag == 1]

        baskets = pd.concat([nonfraudulent_baskets.iloc[:4000], fraudulent_baskets.iloc[:1000]])
        train_baskets, _ = train_test_split(baskets, test_size=TestPipeline.TEST_SIZE, random_state=seed)

        additional_env_variables = {
            "data": train_baskets[["ID"]],
            "labels": train_baskets["fraud_flag"],
            "products": dataset["products"],
        }

        return _pipeline(), additional_env_variables


def _pipeline() -> skrub.DataOp:
    products = skrub.var("products")
    baskets = skrub.var("data")
    labels = skrub.var("labels")

    baskets = baskets.skb.mark_as_X().skb.set_description("Potentially fraudulent shopping baskets of products")
    labels = labels.skb.mark_as_y().skb.set_description(
        "Flag indicating whether the basket is fraudulent (1) or not (0)"
    )

    aggregated = baskets.sem_agg_features(
        products,
        left_on="ID",
        right_on="basket_ID",
        nl_prompt="""
            Generate product features that are indicative of potentially fraudulent baskets, make it easy to distinguish
            anomalous baskets from regular ones! It might be helpful to combine different product statistics.
            """,
        name=TestPipeline.OPERATOR_NAME,
        how_many=1,
    )

    encoded = aggregated.skb.apply(TableVectorizer())
    return encoded.skb.apply(HistGradientBoostingClassifier(random_state=0), y=labels)
