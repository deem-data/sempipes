import warnings

import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

from experiments.colopro import TestPipeline

warnings.filterwarnings("ignore")


class HealthInsurancePipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "insurance"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y_true, y_pred)

    @property
    def scoring(self) -> str:
        return "accuracy"

    def pipeline_with_all_data(self, seed) -> DataOp:
        data = pd.read_csv("experiments/colopro/insurance.csv")
        X = data.iloc[:1000]
        return _pipeline(X)

    def pipeline_with_train_data(self, seed) -> DataOp:
        data = pd.read_csv("experiments/colopro/insurance.csv")
        X = data.iloc[:1000]
        X_train, _ = train_test_split(X, test_size=TestPipeline.TEST_SIZE, random_state=seed)
        return _pipeline(X)


def _pipeline(data) -> skrub.DataOp:
    records = skrub.var("records", data)
    records.skb.set_description("""
        Raw data for insurance Lead Prediction. A policy is recommended to a person when they land on an insurance website, and if the person chooses to fill up a form to apply, it is considered a Positive outcome (Classified as lead). All other conditions are considered Zero outcomes.
    """)
    labels = records["Response"]
    labels.skb.set_description("""
        Insurance policy is recommended to a person when Response is 1.
    """)
    records = records.drop("Response", axis=1)

    records = records.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    records_with_additional_features = records.sem_gen_features(
        nl_prompt="""
            Compute additional features that could help predict whether a person will apply for an insurance policy based on their demographics, and other relevant data. Consider factors such as age, income, and previous insurance history to derive meaningful features that enhance the predictive power of the model.
        """,
        name=TestPipeline.OPERATOR_NAME,
        how_many=1,
    )

    encoded_responses = records_with_additional_features.skb.apply(TableVectorizer())
    return encoded_responses.skb.apply(HistGradientBoostingClassifier(random_state=0), y=labels)
