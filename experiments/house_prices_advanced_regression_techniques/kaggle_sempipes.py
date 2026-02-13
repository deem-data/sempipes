import random
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sempipes  # pylint: disable=unused-import
from experiments.house_prices_advanced_regression_techniques._sempipes_impl import rmsle, sempipes_pipeline

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 0.0},
    ),
)

pipeline = sempipes_pipeline()

data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    random.seed(seed)

    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    train, test = train_test_split(data, test_size=0.5, random_state=seed)

    env_train = pipeline.skb.get_data()
    env_train["data"] = train

    env_test = pipeline.skb.get_data()
    env_test["data"] = test

    learner.fit(env_train)
    y_pred = learner.predict(env_test)
    score = rmsle(np.log1p(test["SalePrice"]), y_pred)
    print(f"RMSLE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
