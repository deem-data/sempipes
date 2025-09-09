import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_log_error

import gyyre  # pylint: disable=unused-import

warnings.filterwarnings("ignore", category=UserWarning)


data = pd.read_csv("comparisons/tmdb-box-office-prediction/train.csv")
revenue_df = data.revenue
data = data.drop(columns=["revenue"])

movie_stats = skrub.var("movie_stats", data).skb.mark_as_X().skb.subsample(n=100)
revenue = skrub.var("revenue", revenue_df).skb.mark_as_y().skb.subsample(n=100)


movie_stats = movie_stats.with_sem_features(
    nl_prompt="""
        Create additional features that could help predict the box office revenue of a movie.
        Consider aspects like genre, production details, cast, crew, and any other relevant information
        that could influence a movie's financial success. Some of the attributes are in JSON format,
        so you might need to parse them to extract useful information.
    """,
    name="additional_movie_features",
    how_many=25,
)

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


def cleanup_column_names(df):
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
    return df


movie_stats = movie_stats.drop(columns=json_columns)

encoder = skrub.TableVectorizer()

X = movie_stats.skb.apply(encoder)
X = X.fillna(-999)

X = X.skb.apply_func(cleanup_column_names)

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
