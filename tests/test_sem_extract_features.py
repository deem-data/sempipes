import skrub
from sklearn.model_selection import cross_validate

import gyyre  # pylint: disable=unused-import


def test_sem_extract_features_text():
    # Fetch a dataset
    dataset = skrub.datasets.fetch_toxicity()
    toxicity = dataset.X[:200]
    binary_label = dataset.y[:200] == "Toxic"

    toxicity_ref = skrub.var("toxicity_text", toxicity)

    # Define the target columns
    toxicity_ref = toxicity_ref.sem_extract_features(
        nl_prompt="Extract a single feature called `hate_speech_presence` from the text tweets that could help predict toxicity.",
        input_cols=["text"],
    ).skb.eval()

    accuracy = (binary_label == toxicity_ref["hate_speech_presence"]).sum() / toxicity.shape[0]
    print(accuracy)
    assert accuracy > 0.5


def test_sem_extract_features_text_explicit_cols():
    # Fetch a dataset
    dataset = skrub.datasets.fetch_toxicity()
    toxicity = dataset.X[:200]
    binary_label = dataset.y[:200] == "Toxic"

    toxicity_ref = skrub.var("toxicity_text", toxicity)

    output_cols = {
        "sentiment_label": "Classify the overall sentiment of the tweet as 'positive', 'negative', or 'neutral.",
        "hate_speech_presence": "Classify with '1' if a tween is toxic and '0' if there is not toxic.",
    }

    # Define the target columns
    toxicity_ref = toxicity_ref.sem_extract_features(
        nl_prompt="Extract features from textual tweets that could help predict toxicity.",
        input_cols=["text"],
        output_cols=output_cols,
    ).skb.eval()

    toxicity_ref["hate_speech_presence"] = toxicity_ref["hate_speech_presence"].astype(int)

    accuracy = (binary_label == toxicity_ref["hate_speech_presence"]).sum() / toxicity.shape[0]
    print(accuracy)
    assert accuracy > 0.5


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
        nl_prompt="Extract up to five features from the text tweets that could help predict toxicity. Focus on sentiment, presence of hate speech, and any other relevant linguistic features. If neutral or not valid content like a link, treat as a no sentiment.",
        input_cols=["text"],
    ).skb.eval()

    model2 = skrub.tabular_pipeline("classifier")
    results2 = cross_validate(model2, toxicity_ref, label)
    print(f"Tabular predictor performance with extracted features: {results2["test_score"]}")

    assert (results["test_score"] < results2["test_score"]).all()


test_sem_extract_features_text_explicit_cols()
