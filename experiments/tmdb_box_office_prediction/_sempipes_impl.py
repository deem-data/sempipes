import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore", category=UserWarning)


def sempipes_pipeline(data_file):
    data = pd.read_csv(data_file)
    revenue_df = data.revenue
    data = data.drop(columns=["revenue"])

    movie_stats = skrub.var("movie_stats", data).skb.mark_as_X().skb.subsample(n=100)
    movie_stats = movie_stats.skb.set_description("""
    In a world… where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate." In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
    """)

    revenue = skrub.var("revenue", revenue_df).skb.mark_as_y().skb.subsample(n=100)
    revenue = revenue.skb.set_description("the international box office revenue for a movie")

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

    def exp_if_transform(outputs, mode=skrub.eval_mode()):
        if mode in {"transform", "predict"}:
            return np.expm1(outputs)
        return outputs

    return predictions.skb.apply_func(exp_if_transform)
