import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_log_error

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    data = pd.read_csv("comparisons/tmdb_box_office_prediction/data.csv")
    train_df, test_df = sklearn.model_selection.train_test_split(data, test_size=0.5, random_state=seed)

    all_df = pd.concat([train_df.drop(["revenue"], axis=1), test_df], axis=0)

    # TODO add code for features

    X_train = all_df[: len(train_df)]
    X_test = all_df[len(train_df) :]
    y_train_log = np.log1p(train_df["revenue"])

    # TODO add model training and prediction
    final_predictions = ...  # TODO Replace with actual predictions

    rmsle = np.sqrt(mean_squared_log_error(test_df["revenue"], final_predictions))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
