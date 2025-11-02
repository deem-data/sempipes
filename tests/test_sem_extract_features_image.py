import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import


def test_sem_extract_features_image_code():
    styles_df = pd.read_csv("tests/data/fashion-dataset/styles.csv", on_bad_lines="skip")
    styles = skrub.var("styles", styles_df)

    X_columns = ["gender", "season", "year", "productDisplayName", "baseColour", "usage"]
    y_column = "masterCategory"

    # Train over texts
    X = styles[X_columns].skb.mark_as_X()
    y = styles[y_column].skb.mark_as_y()

    pipeline = skrub.tabular_pipeline("classifier")
    task = X.skb.apply(pipeline, y=y)
    split = task.skb.train_test_split(random_state=0)
    learner = task.skb.make_learner()
    learner.fit(split["train"])
    score_before = learner.score(split["test"])

    print(f"Tabular predictor performance w/o extracted features: {score_before}")

    # Extract pictures
    styles_full = skrub.var("styles_full", styles_df)
    X_with_features = styles_full[X_columns + ["full_path"]]
    y_new = styles_full[y_column].skb.mark_as_y()

    X_with_features = X_with_features.sem_extract_features(
        nl_prompt="Extract up to three features from the product image and/or the product display name that can be used for the product master category prediction. The features should be very fine-grained and helpful.",
        input_columns=["productDisplayName", "full_path"],
        generate_via_code=True,
    )

    X_with_features = X_with_features.skb.drop(["full_path"])
    X_with_features = X_with_features.skb.mark_as_X()

    pipeline_with_features = skrub.tabular_pipeline("classification")
    task_with_features = X_with_features.skb.apply(pipeline_with_features, y=y_new)
    split_with_features = task_with_features.skb.train_test_split(random_state=0)
    learner_with_features = task_with_features.skb.make_learner()
    learner_with_features.fit(split_with_features["train"])
    score_with_features = learner_with_features.score(split_with_features["test"])
    print(f"Tabular predictor performance with extracted features: {score_with_features}")

    assert score_before <= score_with_features


def test_sem_extract_features_image():
    styles_df = pd.read_csv("tests/data/fashion-dataset/styles.csv", on_bad_lines="skip")
    styles = skrub.var("styles", styles_df)

    X_columns = ["gender", "season", "year", "productDisplayName", "baseColour", "usage"]
    y_column = "masterCategory"

    # Train over texts
    X = styles[X_columns].skb.mark_as_X()
    y = styles[y_column].skb.mark_as_y()

    pipeline = skrub.tabular_pipeline("classifier")
    task = X.skb.apply(pipeline, y=y)
    split = task.skb.train_test_split(random_state=0)
    learner = task.skb.make_learner()
    learner.fit(split["train"])
    score_before = learner.score(split["test"])

    print(f"Tabular predictor performance w/o extracted features: {score_before}")

    # Extract pictures
    styles_full = skrub.var("styles_full", styles_df)
    X_with_features = styles_full[X_columns + ["full_path"]]
    y = styles_full[y_column].skb.mark_as_y()

    X_with_features = X_with_features.sem_extract_features(
        nl_prompt="Extract up to three features from the product image and/or the product display name that can be used for the product master category prediction. The features should be very fine-grained and helpful.",
        input_columns=["productDisplayName", "full_path"],
    )

    X_with_features = X_with_features.skb.drop(["full_path"])
    X_with_features = X_with_features.skb.mark_as_X()

    pipeline_with_features = skrub.tabular_pipeline("classification")
    task_with_features = X_with_features.skb.apply(pipeline_with_features, y=y)
    split_with_features = task_with_features.skb.train_test_split(random_state=0)
    learner_with_features = task_with_features.skb.make_learner()
    learner_with_features.fit(split_with_features["train"])
    score_with_features = learner_with_features.score(split_with_features["test"])
    print(f"Tabular predictor performance with extracted features: {score_with_features}")

    assert score_before <= score_with_features
