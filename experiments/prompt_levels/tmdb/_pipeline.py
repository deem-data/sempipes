import warnings
from typing import Literal

import numpy as np
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")

LearnerKind = Literal["default", "fast"]

DEFAULT_TMDB_FEATURES_PROMPT = """
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
       
        """

MOVIE_STATS_DESCRIPTION = """
        In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
    """

TO_REMOVE_AFTER_FEATURES = [
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


def sempipes_pipeline(
    movie_features_prompt: str | None = None,
    pipeline_seed: int = 42,
    learner: LearnerKind = "default",
):
    movie_features_prompt = movie_features_prompt or DEFAULT_TMDB_FEATURES_PROMPT

    movie_stats = skrub.var("movie_stats").skb.mark_as_X().skb.subsample(n=100)
    movie_stats = movie_stats.skb.set_description(MOVIE_STATS_DESCRIPTION)

    revenue = skrub.var("revenue").skb.mark_as_y().skb.subsample(n=100)
    revenue = revenue.skb.set_description("the international box office revenue for a movie")

    y_log = revenue.skb.apply_func(np.log1p)

    movie_stats = movie_stats.sem_gen_features(
        nl_prompt=movie_features_prompt,
        name="additional_movie_features",
        how_many=30,
    )

    movie_stats = movie_stats.drop(TO_REMOVE_AFTER_FEATURES, axis=1)
    X = movie_stats.skb.apply(skrub.TableVectorizer())

    if learner == "fast":
        selector_estimators, regressor_estimators = 20, 40
    else:
        selector_estimators, regressor_estimators = 40, 100

    feature_selector = SelectFromModel(
        RandomForestRegressor(
            n_estimators=selector_estimators,
            min_samples_leaf=10,
            max_features=0.5,
            n_jobs=-1,
            oob_score=True,
            random_state=42,
        ),
        threshold=0.002,
        importance_getter="feature_importances_",
    )

    X = X.skb.apply(feature_selector, y=y_log)

    regressor = RandomForestRegressor(
        n_estimators=regressor_estimators,
        min_samples_leaf=10,
        max_features=0.5,
        random_state=pipeline_seed,
        n_jobs=-1,
    )

    predictions = X.skb.apply(regressor, y=y_log)

    def exp_if_transform(outputs, mode=skrub.eval_mode()):
        if mode in {"transform", "predict"}:
            return np.expm1(outputs)
        return outputs

    return predictions.skb.apply_func(exp_if_transform)
