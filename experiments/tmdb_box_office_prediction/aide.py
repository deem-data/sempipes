# https://github.com/WecoAI/aideml/blob/main/sample_results/tmdb-box-office-prediction.py
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    all_df = pd.read_csv("experiments/tmdb_box_office_prediction/data.csv")
    train_df, test_df = sklearn.model_selection.train_test_split(all_df, test_size=0.5, random_state=seed)

    train_df["release_date"] = pd.to_datetime(train_df["release_date"], format="%m/%d/%y")
    test_df["release_date"] = pd.to_datetime(test_df["release_date"], format="%m/%d/%y", errors="coerce")

    train_df["release_year"] = train_df["release_date"].dt.year
    train_df["release_month"] = train_df["release_date"].dt.month
    train_df["release_dayofweek"] = train_df["release_date"].dt.dayofweek

    test_df["release_year"] = test_df["release_date"].dt.year
    test_df["release_month"] = test_df["release_date"].dt.month
    test_df["release_dayofweek"] = test_df["release_date"].dt.dayofweek

    test_df["release_year"] = test_df["release_year"].fillna(train_df["release_year"].median())
    test_df["release_month"] = test_df["release_month"].fillna(train_df["release_month"].median())
    test_df["release_dayofweek"] = test_df["release_dayofweek"].fillna(train_df["release_dayofweek"].median())

    features = [
        "budget",
        "popularity",
        "runtime",
        "original_language",
        "release_year",
        "release_month",
        "release_dayofweek",
    ]
    target = "revenue"

    categorical_features = ["original_language"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoded_features = encoder.fit_transform(train_df[categorical_features]).toarray()
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    for feature in features:
        if feature not in categorical_features:
            train_df[feature] = train_df[feature].fillna(train_df[feature].median())

    X = pd.concat(
        [
            train_df[features].drop(columns=categorical_features).reset_index(drop=True),
            encoded_df.reset_index(drop=True),
        ],
        axis=1,
    )
    y = np.log1p(train_df[target])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(verbose=-1, random_state=seed)
    model.fit(X_train, y_train)

    test_encoded_features = encoder.transform(test_df[categorical_features]).toarray()
    test_encoded_df = pd.DataFrame(test_encoded_features, columns=encoded_feature_names)
    for feature in features:
        if feature not in categorical_features:
            test_df[feature] = test_df[feature].fillna(train_df[feature].median())

    X_test = pd.concat(
        [
            test_df[features].drop(columns=categorical_features).reset_index(drop=True),
            test_encoded_df.reset_index(drop=True),
        ],
        axis=1,
    )

    test_pred = model.predict(X_test)
    rmsle = np.sqrt(mean_squared_log_error(test_df["revenue"], np.expm1(test_pred)))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
