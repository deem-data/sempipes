import warnings

import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

from experiments.colopro import TestPipeline

warnings.filterwarnings("ignore")

_DESCRIPTION = """
This dataset contains traffic violation information from all electronic traffic violations issued in the County. 
Any information that can be used to uniquely identify the vehicle, the vehicle owner or the officer issuing the violation will not be published.

The dataset has the following columns:
- seqid: string, several thousand distinct values, 0 missing attributes
- date_of_stop: string, several thousand distinct values, 0 missing attributes
- time_of_stop: string, several thousand distinct values, 0 missing attributes
- agency: string, 1 distinct values, 0 missing attributes
- subagency: string, 9 distinct values, 0 missing attributes
- description: string, several thousand distinct values, some missing attributes

The target is the violation type, which can be 'Warning' or 'Citation':
- violation_type: nominal, 2 distinct values, 0 missing attributes
"""

_TARGET_DESCRIPTION = """
The target is the violation type, which can be 'Warning' or 'Citation'
"""


class TrafficPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "traffic"

    @property
    def scoring(self) -> str:
        return "accuracy"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y_true, y_pred)

    def pipeline_with_all_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_traffic_violations() 
        df = dataset.traffic_violations[dataset.traffic_violations.violation_type.isin(['Warning', 'Citation'])]
        df = df.sample(n=10000, random_state=0)
        X = df.drop(columns=['violation_type'])
        X = X.drop(columns=['search_outcome', 'description', 'charge'])
        y = df['violation_type']

        X_description = _DESCRIPTION
        y_description = _TARGET_DESCRIPTION

        return _pipeline(X, X_description, y, y_description)

    def pipeline_with_train_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_traffic_violations() 
        df = dataset.traffic_violations[dataset.traffic_violations.violation_type.isin(['Warning', 'Citation'])]
        df = df.sample(n=10000, random_state=0)
        X = df.drop(columns=['violation_type'])
        X = X.drop(columns=['search_outcome', 'description', 'charge'])
        y = df['violation_type']

        X_description = _DESCRIPTION
        y_description = _TARGET_DESCRIPTION

        X_train, _, y_train, _ = train_test_split(X, y, train_size=TestPipeline.TEST_SIZE, random_state=seed)

        X_description = dataset.metadata["description"]
        y_description = dataset.metadata["target"]

        return _pipeline(X_train, X_description, y_train, y_description)


def _pipeline(X, X_description, y, y_description) -> skrub.DataOp:
    records = skrub.var("records", X)
    records = records.skb.set_description(X_description)

    labels = skrub.var("labels", y)
    labels = labels.skb.set_name(y_description)

    records = records.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    responses_with_additional_features = records.sem_gen_features(
        nl_prompt="""
            Compute additional features which help predict the traffic violation type.
        """,
        name=TestPipeline.OPERATOR_NAME,
        how_many=5,
    )

    encoded_responses = responses_with_additional_features.skb.apply(TableVectorizer())
    return encoded_responses.skb.apply(RandomForestClassifier(random_state=0), y=labels)
