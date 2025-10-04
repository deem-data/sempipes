import pytest  # pylint: disable=import-error
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

import sempipes

_NUM_SAMPLES = 20


def test_sem_extract_features_text():
    sempipes.update_config(batch_size_for_batch_processing=5)

    dataset = skrub.datasets.fetch_toxicity()
    toxicity = dataset.X[:_NUM_SAMPLES]
    toxicity_labels = dataset.y[:_NUM_SAMPLES] == "Toxic"

    toxicity_ref = skrub.var("toxicity_text", toxicity)

    toxicity_ref = toxicity_ref.sem_extract_features(
        nl_prompt="Extract a single feature called `maybe_toxic` from the text tweets that could help predict toxicity.",
        input_columns=["text"],
    )

    extracted_feature = toxicity_ref["maybe_toxic"].skb.eval()

    accuracy = accuracy_score(toxicity_labels, extracted_feature)
    assert accuracy > 0.5


@pytest.mark.skip(reason="Currently broken with gpt-oss-20b")
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
    accuracy = accuracy_score(toxicity_labels, extracted_feature)
    assert accuracy > 0.5


@pytest.mark.skip(reason="Currently disabled because its too costly")
def test_sem_extract_features_text_pipeline():
    # Fetch a dataset
    dataset = skrub.datasets.fetch_toxicity()
    toxicity = dataset.X
    label = dataset.y
    # Train over texts
    model = skrub.tabular_pipeline("classifier")
    results = cross_validate(model, toxicity, label)
    print(f"Tabular predictor performance w/o extracted features: {results["test_score"]}")

    toxicity_ref = skrub.var("toxicity_text", toxicity)

    # Define the target columns
    toxicity_ref = toxicity_ref.sem_extract_features(
        nl_prompt="Extract up to five features from the text tweets that could help predict toxicity. Focus on sentiment, presence of hate speech, and any other relevant linguistic features. If you encounter neutral or not valid content like a link, treat as a no sentiment.",
        input_columns=["text"],
    ).skb.eval()

    model_with_new_features = skrub.tabular_pipeline("classifier")
    results_with_new_features = cross_validate(model_with_new_features, toxicity_ref, label)
    print(f"Tabular predictor performance with extracted features: {results_with_new_features["test_score"]}")

    assert (results["test_score"] < results_with_new_features["test_score"]).all()
