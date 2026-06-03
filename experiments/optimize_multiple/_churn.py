import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

import sempipes
from experiments.optimize_multiple import TestPipeline

warnings.filterwarnings("ignore")

_OP_CHURN_EXTRACT = "op_churn_extract"
_OP_CHURN_AGGREGATE = "op_churn_aggregate"
_OP_CHURN_NOISE = "op_churn_noise"

_CUSTOMERS_PATH = "experiments/colopro/churn_customers.csv"
_TRANSACTIONS_PATH = "experiments/colopro/churn_transactions.csv"


class ChurnPipeline(TestPipeline):
    OPERATOR_NAMES = [_OP_CHURN_AGGREGATE, _OP_CHURN_EXTRACT, _OP_CHURN_NOISE]

    @property
    def name(self) -> str:
        return "churn"

    def score(self, y_true, y_pred) -> float:
        return accuracy_score(y_true, y_pred)

    @property
    def scoring(self) -> str:
        return "f1"

    def pipeline_with_all_data(self, seed) -> tuple[DataOp, dict]:
        del seed
        all_customers = pd.read_csv(_CUSTOMERS_PATH)
        all_transactions = pd.read_csv(_TRANSACTIONS_PATH)

        all_customers = all_customers[:10000]

        additional_env_variables = {
            "data": all_customers[["CustomerID"]],
            "labels": all_customers["has_churned"],
            "transactions": all_transactions,
        }

        return _pipeline(), additional_env_variables

    def pipeline_with_train_data(self, seed) -> tuple[DataOp, dict]:
        all_customers = pd.read_csv(_CUSTOMERS_PATH)
        all_transactions = pd.read_csv(_TRANSACTIONS_PATH)

        all_customers = all_customers[:10000]
        train_customers, _ = train_test_split(all_customers, test_size=TestPipeline.TEST_SIZE, random_state=seed)

        additional_env_variables = {
            "data": train_customers[["CustomerID"]],
            "labels": train_customers["has_churned"],
            "transactions": all_transactions,
        }
        return _pipeline(), additional_env_variables


def _pipeline() -> DataOp:
    transactions = skrub.var("transactions")

    customer_ids = skrub.var("data")
    churned = skrub.var("labels")

    customer_ids = sempipes.as_X(customer_ids, "Identifiers of customers")
    churned = sempipes.as_y(churned, "Churn status per customer")

    # Weak decoy branches: ID + random seeds only (no transaction access).
    customer_ids_copy = customer_ids.copy(deep=True)
    customer_ids_copy = customer_ids_copy.assign(
        random_col1=lambda df: np.random.rand(len(df)),
        random_col4=lambda df: np.random.choice(["A", "B", "C"], size=len(df)),
    )

    extract_features = customer_ids_copy.sem_extract_features(
        nl_prompt=(
            "Extract features from CustomerID and random_col1 only, using random permutations. "
            "Do not use transaction data. These features are decoys and should not encode churn signal."
        ),
        name=_OP_CHURN_EXTRACT,
        input_columns=["CustomerID", "random_col1"],
        output_columns={
            "random_col2": "Random permutations of values between 0 and 1.",
            "random_col3": "Random penguin species names via permutation.",
        },
    )

    noise_features = customer_ids_copy[["CustomerID", "random_col1", "random_col4"]]
    noise_features = noise_features.sem_gen_features(
        nl_prompt=(
            "Generate at most 2 columns using ONLY random permutations and shuffles of "
            "random_col1 and random_col4. Do not use CustomerID patterns or transaction data. "
            "Features must be pure noise with zero predictive signal for churn."
        ),
        name=_OP_CHURN_NOISE,
        how_many=2,
    )

    # Strong branch: sole access to transaction signal (no leaked aggregates outside this op).
    transactions_for_agg = transactions.rename(columns={"CustomerID": "TransactionCustomerID"})
    aggregated = customer_ids.sem_agg_features(
        transactions_for_agg,
        left_on="CustomerID",
        right_on="TransactionCustomerID",
        nl_prompt="""
            Compute churn-predictive features from shopping transactions per customer:
            transaction count, total spend, average basket size, days since last purchase,
            purchase frequency, and category diversity. These are the primary signal for churn.
            """,
        name=_OP_CHURN_AGGREGATE,
        how_many=5,
    )

    aggregated = aggregated.merge(extract_features, on="CustomerID")
    aggregated = aggregated.merge(noise_features, on="CustomerID")
    encoded = aggregated.skb.apply(TableVectorizer())
    return encoded.skb.apply(
        HistGradientBoostingClassifier(max_depth=4, max_iter=100, random_state=0),
        y=churned,
    )
