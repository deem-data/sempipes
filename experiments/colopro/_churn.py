import warnings

import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

import sempipes
from experiments.colopro import TestPipeline

warnings.filterwarnings("ignore")


class ChurnPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "churn"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y_true, y_pred)

    @property
    def scoring(self) -> str:
        return "accuracy"

    def pipeline_with_all_data(self, seed) -> DataOp:
        all_customers = pd.read_csv("experiments/colopro/churn_customers.csv")
        all_transactions = pd.read_csv("experiments/colopro/churn_transactions.csv")

        all_customers = all_customers[:10000]
        return _pipeline(all_customers, all_transactions)

    def pipeline_with_train_data(self, seed) -> DataOp:
        all_customers = pd.read_csv("experiments/colopro/churn_customers.csv")
        all_transactions = pd.read_csv("experiments/colopro/churn_transactions.csv")

        all_customers = all_customers[:10000]
        train_customers, _ = train_test_split(all_customers, test_size=TestPipeline.TEST_SIZE, random_state=seed)
        return _pipeline(train_customers, all_transactions)


def _pipeline(labeled_customers, customer_transactions) -> skrub.DataOp:
    customer_ids = skrub.var("customers", labeled_customers[["CustomerID"]])
    churned = skrub.var("churned", labeled_customers["has_churned"])
    transactions = skrub.var("transactions", customer_transactions)

    customer_ids = sempipes.as_X(customer_ids, "Identifiers of customers")
    churned = sempipes.as_y(churned, "Churn status per customer")

    num_transactions = transactions.groupby("CustomerID").size().reset_index(name="num_transactions")
    customer_ids = customer_ids.merge(num_transactions, on="CustomerID", how="left")
    customer_ids = customer_ids.assign(num_transactions=lambda df: df["num_transactions"].fillna(0).astype(int))

    aggregated = customer_ids.sem_agg_features(
        transactions,
        left_on="CustomerID",
        right_on="CustomerID",
        nl_prompt="""
            Compute features from the shopping transactions that indicate whether a customer will buy something in 
            the following years or not.
            """,
        name=TestPipeline.OPERATOR_NAME,
        how_many=1,
    )

    encoded = aggregated.skb.apply(TableVectorizer())
    return encoded.skb.apply(HistGradientBoostingClassifier(random_state=0), y=churned)
