import warnings

import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

from experiments.colopro import TestPipeline

warnings.filterwarnings("ignore")


class MidwestSurveyPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "midwestsurvey"

    @property
    def scoring(self) -> str:
        return "accuracy"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y_true, y_pred)

    def pipeline_with_all_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_midwest_survey()

        X = dataset.X.iloc[:2000]
        mask = ~X["In_what_ZIP_code_is_your_home_located"].str.contains(r"[^0-9.]")
        X = X[mask]
        y = dataset.y.iloc[:2000][mask]

        X_description = dataset.metadata["description"]
        y_description = dataset.metadata["target"]

        return _pipeline(X, X_description, y, y_description)

    def pipeline_with_train_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_midwest_survey()

        X = dataset.X.iloc[:2000]
        mask = ~X["In_what_ZIP_code_is_your_home_located"].str.contains(r"[^0-9.]")
        X = X[mask]
        y = dataset.y.iloc[:2000][mask]

        X_train, _, y_train, _ = train_test_split(X, y, train_size=TestPipeline.TEST_SIZE, random_state=seed)

        X_description = dataset.metadata["description"]
        y_description = dataset.metadata["target"]

        return _pipeline(X_train, X_description, y_train, y_description)


def _pipeline(X, X_description, y, y_description) -> skrub.DataOp:
    responses = skrub.var("response", X)
    responses = responses.skb.set_description(X_description)

    labels = skrub.var("labels", y)
    labels = labels.skb.set_name(y_description)

    responses = responses.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    responses_with_additional_features = responses.sem_gen_features(
        nl_prompt="""
            Compute additional features which help predict the census region of a respondent based on their demographics. Use your intrinsic knowledge about the US to come up with the features. Pay special attention to the zip code of the person.
        """,
        name=TestPipeline.OPERATOR_NAME,
        how_many=5,
    )

    encoded_responses = responses_with_additional_features.skb.apply(TableVectorizer())
    return encoded_responses.skb.apply(RandomForestClassifier(random_state=0), y=labels)
