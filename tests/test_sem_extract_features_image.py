import os

import kagglehub  # pylint: disable=import-error
import pandas as pd
import skrub
from sklearn.model_selection import cross_validate

import gyyre  # pylint: disable=unused-import


def test_sem_extract_features_image():
    # Fetch a dataset
    dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    styles_csv_path = os.path.join(dataset_path, "fashion-dataset", "styles.csv")

    styles = pd.read_csv(styles_csv_path, on_bad_lines="skip")[:200]

    # Extract the whole dataset
    styles["full_path"] = [
        os.path.join(dataset_path, "fashion-dataset", "images", f"{img}.jpg") for img in styles["id"]
    ]
    X_cols = ["gender", "season", "year", "productDisplayName", "baseColour", "usage"]
    y_col = "masterCategory"

    # Train over texts
    model = skrub.tabular_pipeline("classifier")
    results = cross_validate(model, styles[X_cols], styles[y_col])
    print(f"Tabular predictor performance w/o extracted features: {results["test_score"]}")

    # Extract pictures
    styles_ref = skrub.var("styles", styles)
    styles_ref = styles_ref.sem_extract_features(
        nl_prompt="Extract up to three features from the product image and/or the product display name that can be used for the product master category prediction. The features should be very fine-grained and helpful.",
        input_columns=["productDisplayName", "full_path"],
    ).skb.eval()

    X_cols_remove = ["subCategory", "id", "full_path", "masterCategory", "articleType"]
    X_cols_with_new_features = styles.columns[~styles.columns.isin(X_cols_remove)]

    model_with_new_features = skrub.tabular_pipeline("classifier")
    results_with_new_features = cross_validate(
        model_with_new_features, styles_ref[X_cols_with_new_features], styles[y_col]
    )
    print(f"Tabular predictor performance with extracted features: {results_with_new_features["test_score"]}")

    assert (results_with_new_features["test_score"] >= results["test_score"]).all()


test_sem_extract_features_image()
