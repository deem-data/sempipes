import math

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skrub import selectors as s

import sempipes
from sempipes.optimisers.trajectory import load_trajectory_from_json

trajectory = load_trajectory_from_json("experiments/house_prices_advanced_regression_techniques/swe.json")
best_outcome = max(trajectory.outcomes, key=lambda x: (x.score, -x.search_node.trial))


def sempipes_pipeline():
    data = skrub.var("data").skb.mark_as_X()
    y = data["SalePrice"]
    data = data.drop(columns=["SalePrice", "Id"])

    y = y.skb.apply_func(np.log).skb.mark_as_y()

    data = data.sem_gen_features(
        nl_prompt="""
        Create additional features that could help predict the sale price of a house.
        Consider aspects like location, size, condition, and any other relevant information.
        You way want to combine several existing features to create new ones.
        """,
        name="house_features",
        how_many=10,
    )

    data = data.skb.apply(SimpleImputer(strategy="median"), cols=s.numeric())
    data = data.skb.apply(SimpleImputer(strategy="most_frequent"), cols=s.categorical())

    data = data.skb.apply(skrub.TableVectorizer())
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    predictions = data.skb.apply(model, y=y)
    return predictions


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 0.0},
        ),
    )

    scores = []
    seed = 42
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        train = all_data.iloc[train_idx]
        test = all_data.iloc[test_idx]
        np.random.seed(seed)

        y_true = np.log(test["SalePrice"])

        predictions = sempipes_pipeline()
        learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)

        env_train = predictions.skb.get_data()
        env_train["data"] = train
        env_train["sempipes_prefitted_state__house_features"] = best_outcome.state
        env_test = predictions.skb.get_data()
        env_test["data"] = test
        env_test["sempipes_prefitted_state__house_features"] = best_outcome.state
        learner.fit(env_train)
        y_pred = learner.predict(env_test)

        score = math.sqrt(mean_squared_error(y_true, y_pred))
        print(f"RMSLE on split {fold_index}: {score}")
        scores.append(score)

    print("\nMean final score: ", np.mean(scores), np.std(scores))
