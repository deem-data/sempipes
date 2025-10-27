# Based on https://www.kaggle.com/code/archfu/notebook95778def47
import ast

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OrdinalEncoder

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    data = pd.read_csv("experiments/tmdb_box_office_prediction/data.csv")
    train_df, test_df = sklearn.model_selection.train_test_split(data, test_size=0.5, random_state=seed)

    all_df = pd.concat([train_df.drop(["revenue"], axis=1), test_df], axis=0)

    def parse_json(column_str):
        if isinstance(column_str, str):
            try:
                return ast.literal_eval(column_str)
            except Exception as __e:  # pylint: disable=broad-exception-caught
                return []
        return []

    json_columns = [
        "belongs_to_collection",
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "Keywords",
        "cast",
        "crew",
    ]

    for col in json_columns:
        all_df[col] = all_df[col].apply(parse_json)

    all_df["has_collection"] = all_df["belongs_to_collection"].apply(lambda x: 1 if x else 0)
    all_df["has_homepage"] = all_df["homepage"].notna().astype(int)
    all_df["has_tagline"] = all_df["tagline"].notna().astype(int)
    all_df["budget"] = all_df["budget"].replace(0, np.nan)
    all_df["budget"] = all_df["budget"].fillna(train_df["budget"].median())
    all_df["runtime"] = all_df["runtime"].fillna(train_df["runtime"].median())

    for col in (
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "Keywords",
        "cast",
        "crew",
    ):
        all_df[f"num_{col}"] = all_df[col].apply(len)

    def get_names_from_list(data_list, key="name", top_n=5):
        if not isinstance(data_list, list):
            return ["Unknown"] * top_n
        names = [item.get(key, "Unknown") for item in data_list]
        return (names + ["Unknown"] * top_n)[:top_n]

    def get_job_name(crew_list, job_title):
        if not isinstance(crew_list, list):
            return "Unknown"
        return next((member["name"] for member in crew_list if member.get("job") == job_title), "Unknown")

    all_df["first_genre"] = all_df["genres"].apply(lambda x: x[0]["name"] if x else "Unknown")
    all_df["production_company"] = all_df["production_companies"].apply(lambda x: x[0]["name"] if x else "Unknown")
    all_df["director"] = all_df["crew"].apply(get_job_name, job_title="Director")
    all_df["writer"] = all_df["crew"].apply(get_job_name, job_title="Writer")
    all_df["producer"] = all_df["crew"].apply(get_job_name, job_title="Producer")
    all_df["lead_actor_name"] = all_df["cast"].apply(lambda x: x[0]["name"] if x else "Unknown")
    all_df["overview_word_count"] = all_df["overview"].str.split().str.len().fillna(0)
    all_df["title_char_count"] = all_df["title"].str.len().fillna(0)

    categorical_cols = [
        "first_genre",
        "production_company",
        "director",
        "writer",
        "producer",
        "lead_actor_name",
        "original_language",
    ]
    for col in categorical_cols:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",  # map unseen categories to unknown_value
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int64,
        )
        train_range = all_df[: len(train_df)]
        test_range = all_df[len(train_df) :]
        encoded_train_values = ordinal_encoder.fit_transform(train_range[[col]])
        encoded_test_values = ordinal_encoder.transform(test_range[[col]])
        all_df[f"{col}_code"] = np.concatenate([encoded_train_values, encoded_test_values])

    train_processed = all_df[: len(train_df)]
    test_processed = all_df[len(train_df) :]

    features = [
        "budget",
        "popularity",
        "runtime",
        "has_collection",
        "has_homepage",
        "has_tagline",
        "num_genres",
        "num_production_companies",
        "num_production_countries",
        "num_spoken_languages",
        "num_Keywords",
        "num_cast",
        "num_crew",
        "overview_word_count",
        "title_char_count",
        "first_genre_code",
        "production_company_code",
        "director_code",
        "writer_code",
        "producer_code",
        "lead_actor_name_code",
        "original_language_code",
    ]

    X_train = train_processed[features]
    X_test = test_processed[features]
    y_train_log = np.log1p(train_df["revenue"])

    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)
    params = {
        "objective": "regression_l1",
        "metric": "rmse",
        "n_estimators": 3000,
        "learning_rate": 0.003,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "boosting_type": "gbdt",
    }
    model = lgb.LGBMRegressor(**params)  # type: ignore
    model.fit(X_train, y_train_log)

    log_predictions = model.predict(X_test)
    final_predictions = np.expm1(log_predictions)

    from sklearn.metrics import mean_squared_log_error

    rmsle = np.sqrt(mean_squared_log_error(test_df["revenue"], final_predictions))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
