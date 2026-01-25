import warnings

import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

from experiments.colopro import TestPipeline

warnings.filterwarnings("ignore")


class TweetsPipeline(TestPipeline):
    @property
    def name(self) -> str:
        return "tweets"

    @property
    def scoring(self) -> str:
        return "accuracy"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y_true, y_pred)

    def pipeline_with_all_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_toxicity()

        X = dataset.X.iloc[:600]
        y = dataset.y.iloc[:600] == "Toxic"

        env_variables = {
            "data": X,
            "labels": y,
        }

        return _pipeline(), env_variables

    def pipeline_with_train_data(self, seed) -> DataOp:
        dataset = skrub.datasets.fetch_toxicity()

        X = dataset.X.iloc[:600]
        y = dataset.y.iloc[:600] == "Toxic"

        X_train, _, y_train, _ = train_test_split(X, y, train_size=TestPipeline.TEST_SIZE, random_state=seed)

        env_variables = {
            "data": X_train,
            "labels": y_train,
        }

        return _pipeline(), env_variables


def _pipeline() -> skrub.DataOp:
    tweets = skrub.var("data")
    labels = skrub.var("labels")

    tweets = tweets.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    output_columns = {
        "maybe_toxic": "Respond with '1' if a tweet is toxic and '0' if its text is not toxic.",
        "sentiment_label": "Classify the overall sentiment of the tweet as 'positive', 'negative', or 'neutral.",
    }

    labeled_tweets = tweets.sem_extract_features(
        nl_prompt="Extract features from textual tweets that could help predict toxicity.",
        input_columns=["text"],
        name=TestPipeline.OPERATOR_NAME,
        output_columns=output_columns,
        generate_via_code=True,
        print_code_to_console=True,
    )

    encoded_tweets = labeled_tweets.skb.apply(TableVectorizer())
    return encoded_tweets.skb.apply(RandomForestClassifier(random_state=0), y=labels)
