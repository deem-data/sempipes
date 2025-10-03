import pandas as pd
import skrub
from sklearn.model_selection import cross_validate

import sempipes  # pylint: disable=unused-import


def test_sem_extract_features_image():
    styles = pd.read_csv("tests/data/fashion-dataset/styles.csv", on_bad_lines="skip")

    X_columns = ["gender", "season", "year", "productDisplayName", "baseColour", "usage"]
    y_column = "masterCategory"

    # Train over texts
    model = skrub.tabular_pipeline("classifier")
    results = cross_validate(model, styles[X_columns], styles[y_column])
    print(f"Tabular predictor performance w/o extracted features: {results["test_score"]}")

    # Extract pictures
    styles_ref = skrub.var("styles", styles)
    styles_ref = styles_ref.sem_extract_features(
        nl_prompt="Extract up to three features from the product image and/or the product display name that can be used for the product master category prediction. The features should be very fine-grained and helpful.",
        input_columns=["productDisplayName", "full_path"],
    ).skb.eval()

    X_columns_to_remove = ["subCategory", "id", "full_path", "masterCategory", "articleType"]
    X_columns_with_new_features = styles.columns[~styles.columns.isin(X_columns_to_remove)]

    model_with_new_features = skrub.tabular_pipeline("classifier")
    results_with_new_features = cross_validate(
        model_with_new_features, styles_ref[X_columns_with_new_features], styles[y_column]
    )
    print(f"Tabular predictor performance with extracted features: {results_with_new_features["test_score"]}")

    assert (results_with_new_features["test_score"] >= results["test_score"]).all()


test_sem_extract_features_image()
