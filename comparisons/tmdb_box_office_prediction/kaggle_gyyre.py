import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_log_error

import gyyre  # pylint: disable=unused-import

warnings.filterwarnings("ignore", category=UserWarning)


data = pd.read_csv("comparisons/tmdb_box_office_prediction/train.csv")
revenue_df = data.revenue
data = data.drop(columns=["revenue"])

print("\n\n\n\n\n\n\n\n\ndata")
print(data)
print(data.columns)

movie_stats = skrub.var("movie_stats", data).skb.mark_as_X().skb.subsample(n=100)
movie_stats = movie_stats.skb.set_description("""
In a world… where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate." In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
""")

revenue = skrub.var("revenue", revenue_df).skb.mark_as_y().skb.subsample(n=100)
revenue = revenue.skb.set_description("the international box office revenue for a movie")


movie_stats = movie_stats.assign(spoken_languages_str=movie_stats["spoken_languages"].apply(json.dumps))
movie_stats = movie_stats.assign(cast_str=movie_stats["cast"].apply(json.dumps))
movie_stats = movie_stats.sem_extract_features(
    nl_prompt="""
        Extract FOUR helpful features that are very helpful for the movie revenue prediction. 
        Extract new features from the movie overview, spoken languages, and cast columns.
        Consider that genre spoken languages and cast columns are in JSON format.
    """,
    input_columns=["overview", "spoken_languages_str", "cast_str"],
)


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

print("\n\n\n\n\n\n\n\n\nmovie_stats")
print(movie_stats.columns)

json_columns = [
    "belongs_to_collection",
    "genres",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "spoken_languages_str",
    "Keywords",
    "cast",
    "cast_str",
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

print("\n\n\n\n\n\n\n\ncolumns")
print(movie_stats.columns)

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = predictions.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)
    print(split["train"])
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])

    rmsle = np.sqrt(mean_squared_log_error(split["test"]["_skrub_y"], y_pred))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
