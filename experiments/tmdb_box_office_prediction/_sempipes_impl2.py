import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")


def sempipes_pipeline2(data_file: str, pipeline_seed):
    data = pd.read_csv(data_file)
    revenue_df = data.revenue
    data = data.drop(columns=["revenue"])

    movie_stats = skrub.var("movie_stats", data).skb.mark_as_X().skb.subsample(n=100)
    movie_stats = movie_stats.skb.set_description("""
        In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
    """)

    revenue = skrub.var("revenue", revenue_df).skb.mark_as_y().skb.subsample(n=100)
    revenue = revenue.skb.set_description("the international box office revenue for a movie")

    y_log = revenue.skb.apply_func(np.log1p)

    movie_stats = movie_stats.sem_gen_features(
        nl_prompt="""
        Create additional features that could help predict the box office revenue of a movie. Here are detailed instructions:

        Compute the year, month, day, day of week, day of year of the movie release data, and the elapsed time since then

        Compute the following features:
        - number of cast members
        - number of crew members
        - whether the movie has a tagline
        - length of the tagline (in words)
        - whether the movie has an overview
        - length of the overview (in words)
        - whether the movie has a budget greater than zero
        - whether the movie has a homepage

        Next, extract the following metadata:
        - director of the movie
        - screenplay writer
        - director of photography
        - original music composer
        - art director       

        Besides that, think creatively about other features that could be useful for predicting box office revenue, 
        and create them as well.
       
        """,
        name="additional_movie_features",
        how_many=30,
    )

    to_remove = [
        "imdb_id",
        "id",
        "poster_path",
        "overview",
        "homepage",
        "tagline",
        "original_title",
        "status",
        "cast",
        "release_date",
        "Keywords",
        "crew",
        "belongs_to_collection",
    ]

    movie_stats = movie_stats.drop(to_remove, axis=1)
    X = movie_stats.skb.apply(skrub.TableVectorizer())

    feature_selector = SelectFromModel(
        RandomForestRegressor(
            n_estimators=40, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True, random_state=42
        ),
        threshold=0.002,
        importance_getter="feature_importances_",
    )

    X = X.skb.apply(feature_selector, y=y_log)

    regressor = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=10, max_features=0.5, random_state=pipeline_seed, n_jobs=-1
    )

    predictions = X.skb.apply(regressor, y=y_log)

    def exp_if_transform(outputs, mode=skrub.eval_mode()):
        if mode in {"transform", "predict"}:
            return np.expm1(outputs)
        return outputs

    return predictions.skb.apply_func(exp_if_transform)
