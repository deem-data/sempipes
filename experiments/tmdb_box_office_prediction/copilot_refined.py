import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    data = pd.read_csv("experiments/tmdb_box_office_prediction/data.csv")
    train_df, test_df = sklearn.model_selection.train_test_split(data, test_size=0.5, random_state=seed)

    all_df = pd.concat([train_df.drop(["revenue"], axis=1), test_df], axis=0)

    # --- Feature Engineering ---
    # Use only numeric columns for simplicity
    numeric_features = ["budget", "popularity", "runtime"]
    # Fill missing values
    for col in numeric_features:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce").fillna(all_df[col].median())
    # Encode original_language as categorical
    all_df["original_language"] = all_df["original_language"].astype("category").cat.codes

    X_train = all_df[: len(train_df)][numeric_features + ["original_language"]]
    X_test = all_df[len(train_df) :][numeric_features + ["original_language"]]
    y_train_log = np.log1p(train_df["revenue"])

    # --- Model Training and Prediction ---
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    final_predictions = np.expm1(y_pred_log)

    rmsle = np.sqrt(mean_squared_log_error(test_df["revenue"], final_predictions))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
