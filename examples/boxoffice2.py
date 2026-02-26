import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

import sempipes

warnings.filterwarnings("ignore")


def sempipes_pipeline():
    movie_stats = skrub.var("data").skb.mark_as_X()
    movie_stats = movie_stats.skb.set_description("""
        In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
    """)

    revenue = movie_stats["revenue"].skb.mark_as_y()
    revenue = revenue.skb.set_description("the international box office revenue for a movie")

    movie_stats = movie_stats.drop(columns=["revenue"])

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
        n_estimators=100, min_samples_leaf=10, max_features=0.5, random_state=42, n_jobs=-1
    )

    predictions = X.skb.apply(regressor, y=y_log)

    def exp_if_transform(outputs, mode=skrub.eval_mode()):
        if mode in {"transform", "predict"}:
            return np.expm1(outputs)
        return outputs

    return predictions.skb.apply_func(exp_if_transform)


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 0.0},
        )
    )

    data = pd.read_csv("examples/data/boxoffice.csv")
    train, test = train_test_split(data, test_size=0.75, random_state=42)

    pipeline = sempipes_pipeline()
    learner = pipeline.skb.make_learner()
    env_train = pipeline.skb.get_data()
    env_train["data"] = train
    learner.fit(env_train)

    env_test = pipeline.skb.get_data()
    env_test["data"] = test
    y_pred = learner.predict(env_test)

    rmsle = np.sqrt(mean_squared_log_error(test["revenue"], y_pred))
    print(f"RMSLE: {rmsle}")
