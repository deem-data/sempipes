import ast

import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv("comparisons/tmdb_box_office_prediction/train.csv")
revenue_df = data.revenue
data = data.drop(columns=["revenue"])

movie_stats = skrub.var("movie_stats", data).skb.mark_as_X().skb.subsample(n=100)
revenue = skrub.var("revenue", revenue_df).skb.mark_as_y().skb.subsample(n=100)


def parse_json(column_str):
    if isinstance(column_str, str):
        try:
            return ast.literal_eval(column_str)
        except Exception as __e:  # pylint: disable=broad-exception-caught
            return []
    return []


movie_stats = movie_stats.assign(
    has_collection=lambda df: df["belongs_to_collection"].notna().astype(int),  # .apply(lambda x: 1 if x else 0),
    has_homepage=lambda df: df["homepage"].notna().astype(int),
    has_tagline=lambda df: df["tagline"].notna().astype(int),
    budget=lambda df: df["budget"].replace(0, np.nan),
)


def nan_safe_len(val_str):
    if pd.isna(val_str):
        return 0
    val = parse_json(val_str)
    return len(val)


movie_stats = movie_stats.assign(
    num_genres=lambda df: df["genres"].apply(nan_safe_len),
    num_production_companies=lambda df: df["production_companies"].apply(nan_safe_len),
    num_production_countries=lambda df: df["production_countries"].apply(nan_safe_len),
    num_spoken_languages=lambda df: df["spoken_languages"].apply(nan_safe_len),
    num_Keywords=lambda df: df["Keywords"].apply(nan_safe_len),
    num_cast=lambda df: df["cast"].apply(nan_safe_len),
    num_crew=lambda df: df["crew"].apply(nan_safe_len),
)


def get_names_from_list(data_list_str, key="name", top_n=5):
    data_list = parse_json(data_list_str)
    if not isinstance(data_list, list):
        return ["Unknown"] * top_n
    names = [item.get(key, "Unknown") for item in data_list]
    return (names + ["Unknown"] * top_n)[:top_n]


def get_job_name(crew_list_str, job_title):
    crew_list = parse_json(crew_list_str)
    if not isinstance(crew_list, list):
        return "Unknown"
    return next((member["name"] for member in crew_list if member.get("job") == job_title), "Unknown")


def get_first_name(x_str):
    x = parse_json(x_str)
    if isinstance(x, list) and len(x) > 0:
        if isinstance(x[0], dict) and "name" in x[0]:
            return x[0]["name"]
    return "Unknown"


movie_stats = movie_stats.assign(
    first_genre=lambda df: df["genres"].apply(get_first_name),
    production_company=lambda df: df["production_companies"].apply(get_first_name),
    director=lambda df: df["crew"].apply(get_job_name, job_title="Director"),
    writer=lambda df: df["crew"].apply(get_job_name, job_title="Writer"),
    producer=lambda df: df["crew"].apply(get_job_name, job_title="Producer"),
    lead_actor_name=lambda df: df["cast"].apply(get_first_name),
    overview_word_count=lambda df: df["overview"].str.split().str.len().fillna(0),
    title_char_count=lambda df: df["title"].str.len().fillna(0),
)

categorical_cols = [
    "first_genre",
    "production_company",
    "director",
    "writer",
    "producer",
    "lead_actor_name",
    "original_language",
]
ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",  # map unseen categories to unknown_value
    unknown_value=-1,
    encoded_missing_value=-1,
    dtype=np.int64,
)

pass_through_features = [
    "popularity",
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
]


encoder = ColumnTransformer(
    transformers=[
        ("imputed", SimpleImputer(strategy="median"), ["budget", "runtime"]),
        ("passthrough", "passthrough", pass_through_features),
        ("ordinal", ordinal_encoder, categorical_cols),
    ]
)

X = movie_stats.skb.apply(encoder)
X = X.fillna(-999)

y_log = revenue.skb.apply_func(np.log1p)

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
predictions = X.skb.apply(model, y=y_log)
predictions = predictions.skb.apply_func(np.expm1)


scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = predictions.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])

    rmsle = np.sqrt(mean_squared_log_error(split["test"]["_skrub_y"], y_pred))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
