import lightgbm as lgb
import numpy as np
import pandas as pd
import sempipes
import skrub
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
        release_dayofweek=lambda x: x["release_date"].dt.dayofweek
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
            parameters={"temperature": 0.0},
        ),
    )

    scores = []
    for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
        print(f"Split {split_index}")
        np.random.seed(seed)
        all_df = pd.read_csv("experiments/tmdb_box_office_prediction/data.csv")
        train_df, test_df = train_test_split(all_df, test_size=0.5, random_state=seed)

        predictions = sempipes_pipeline()
        learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)

        env_train = predictions.skb.get_data()    
        env_train["data"] = train_df
        env_test = predictions.skb.get_data()   
        env_test["data"] = test_df

        learner.fit(env_train)
        y_pred = learner.predict(env_test)

        rmsle = np.sqrt(mean_squared_log_error(test_df["revenue"], np.expm1(y_pred)))
        print(f"RMSLE on split {split_index}: {rmsle}")
        scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))