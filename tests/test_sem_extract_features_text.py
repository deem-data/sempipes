import pytest  # pylint: disable=import-error
import skrub
from sklearn.metrics import accuracy_score

import sempipes

_NUM_SAMPLES = 20


def test_sem_extract_features_text():
    sempipes.update_config(batch_size_for_batch_processing=5)

    dataset = skrub.datasets.fetch_toxicity()
    toxicity = dataset.X[:_NUM_SAMPLES]
    toxicity_labels = dataset.y[:_NUM_SAMPLES] == "Toxic"

    toxicity_ref = skrub.var("toxicity_text", toxicity)

    toxicity_ref = toxicity_ref.sem_extract_features(
        nl_prompt="Extract a single binary feature called `maybe_toxic` from the text tweets that could help predict toxicity.",
        input_columns=["text"],
    )

    extracted_feature = toxicity_ref["maybe_toxic"].skb.eval()

    accuracy = accuracy_score(toxicity_labels, extracted_feature)
    assert accuracy > 0.5


def test_sem_extract_features_text_with_code():
    sempipes.update_config(batch_size_for_batch_processing=5)

    dataset = skrub.datasets.fetch_toxicity()
    toxicity = dataset.X[:_NUM_SAMPLES]
    toxicity_labels = dataset.y[:_NUM_SAMPLES] == "Toxic"

    toxicity_ref = skrub.var("toxicity_text", toxicity)

    toxicity_ref = toxicity_ref.sem_extract_features(
        nl_prompt="Extract a single binary feature called `maybe_toxic` from the text tweets that could help predict toxicity.",
        input_columns=["text"],
        generate_via_code=True,
    )

    extracted_feature = toxicity_ref["maybe_toxic"].skb.eval()

    accuracy = accuracy_score(toxicity_labels, extracted_feature)
    assert accuracy > 0.5


def test_sem_extract_features_text_explicit_cols():
    sempipes.update_config(batch_size_for_batch_processing=5)

    dataset = skrub.datasets.fetch_toxicity()
    toxicity = dataset.X[:_NUM_SAMPLES]
    toxicity_labels = dataset.y[:_NUM_SAMPLES] == "Toxic"

    toxicity_ref = skrub.var("toxicity_text", toxicity)

    output_columns = {
        "sentiment_label": "Classify the overall sentiment of the tweet as 'positive', 'negative', or 'neutral.",
        "maybe_toxic": "Respond with '1' if a tweet is toxic and '0' if its text is not toxic.",
    }

    toxicity_ref = toxicity_ref.sem_extract_features(
        nl_prompt="Extract features from textual tweets that could help predict toxicity.",
        input_columns=["text"],
        output_columns=output_columns,
    )

    extracted_feature = toxicity_ref["maybe_toxic"].skb.eval()
    extracted_feature = extracted_feature.astype(int)
    accuracy = accuracy_score(toxicity_labels, extracted_feature)
    assert accuracy > 0.5


def test_sem_extract_features_text_pipeline_code():
    # Fetch a dataset
    dataset_df = skrub.datasets.fetch_toxicity()
    dataset = skrub.var("toxicity", dataset_df)

    X = dataset.X.skb.mark_as_X()
    y = dataset.y.skb.mark_as_y()

    pipeline = skrub.tabular_pipeline("classification")
    task = X.skb.apply(pipeline, y=y)
    split = task.skb.train_test_split(random_state=0)
    learner = task.skb.make_learner()
    learner.fit(split["train"])
    score_before = learner.score(split["test"])
    print(f"Tabular predictor performance w/o extracted features: {score_before}")

    # Define the target columns
    X = X.sem_extract_features(
        nl_prompt="Extract up to five features from the text tweets that could help predict toxicity. Focus on sentiment, presence of hate speech, and any other relevant linguistic features. If you encounter neutral or not valid content like a link, treat as a no sentiment.",
        input_columns=["text"],
        generate_via_code=True,
    )

    pipeline_with_features = skrub.tabular_pipeline("classification")
    task_with_features = X.skb.apply(pipeline_with_features, y=y)
    split_with_features = task_with_features.skb.train_test_split(random_state=0)
    learner_with_features = task.skb.make_learner()
    learner_with_features.fit(split_with_features["train"])
    score_with_features = learner.score(split_with_features["test"])
    print(f"Tabular predictor performance with extracted features: {score_with_features}")

    assert score_before <= score_with_features


@pytest.mark.skip(reason="Currently disabled because its too costly")
def test_sem_extract_features_text_pipeline():
    # Fetch a dataset
    dataset_df = skrub.datasets.fetch_toxicity()
    dataset = skrub.var("toxicity", dataset_df)

    X = dataset.X.skb.mark_as_X()
    y = dataset.y.skb.mark_as_y()

    pipeline = skrub.tabular_pipeline("classification")
    task = X.skb.apply(pipeline, y=y)
    split = task.skb.train_test_split(random_state=0)
    learner = task.skb.make_learner()
    learner.fit(split["train"])
    score_before = learner.score(split["test"])
    print(f"Tabular predictor performance w/o extracted features: {score_before}")

    # Define the target columns
    X_with_features = X.sem_extract_features(
        nl_prompt="Extract up to five features from the text tweets that could help predict toxicity. Focus on sentiment, presence of hate speech, and any other relevant linguistic features. If you encounter neutral or not valid content like a link, treat as a no sentiment.",
        input_columns=["text"],
    ).skb.eval()

    pipeline_with_features = skrub.tabular_pipeline("classification")
    task_with_features = X_with_features.skb.apply(pipeline_with_features, y=y)
    split_with_features = task_with_features.skb.train_test_split(random_state=0)
    learner_with_features = task.skb.make_learner()
    learner_with_features.fit(split_with_features["train"])
    score_with_features = learner.score(split_with_features["test"])
    print(f"Tabular predictor performance with extracted features: {score_with_features}")

    assert score_before < score_with_features
