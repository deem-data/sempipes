import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
from sklearn.impute import SimpleImputer

import sempipes
from sempipes.optimisers import optimise_colopro
from sempipes.optimisers.evolutionary_search import EvolutionarySearch


def sempipes_pipeline():
    data = skrub.var("data").skb.mark_as_X()

    y = data["revenue"].skb.apply_func(np.log1p).skb.mark_as_y()
    data = data.drop(columns=["revenue"])

    def cast_release_date(df):
        df["release_date"] = pd.to_datetime(df["release_date"], format="%m/%d/%y")
        return df

    data = data.skb.apply_func(cast_release_date)
    data = data.assign(
        release_year=lambda x: x["release_date"].dt.year,
        release_month=lambda x: x["release_date"].dt.month,
        release_dayofweek=lambda x: x["release_date"].dt.dayofweek,
    )

    data = data.skb.apply(SimpleImputer(strategy="median"), cols=["release_year", "release_month", "release_dayofweek"])

    X = data.sem_gen_features(
        nl_prompt="""
            Create additional features that could help predict the box office revenue of a movie.
            Consider aspects like genre, production details, cast, crew, and any other relevant information
            that could influence a movie's financial success. Some of the attributes are in JSON format,
            so you might need to parse them to extract useful information.
        """,
        name="additional_movie_features",
        how_many=25,
    )

    X = X.skb.apply(skrub.TableVectorizer())
    model = lgb.LGBMRegressor(verbose=-1, random_state=42)

    predictions = X.skb.apply(model, y=y)
    return predictions


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
    )

    validation_data = pd.read_csv("experiments/tmdb_box_office_prediction/validation.csv")

    pipeline = sempipes_pipeline()

    outcomes = optimise_colopro(
        pipeline,
        "additional_movie_features",
        num_trials=24,
        search=EvolutionarySearch(population_size=6),
        scoring="neg_root_mean_squared_error",
        cv=10,
        additional_env_variables={"data": validation_data},
    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print("\n".join(best_outcome.state["generated_code"]))
