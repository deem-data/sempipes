import json
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold, train_test_split

import sempipes  # pylint: disable=unused-import
from experiments.house_prices_advanced_regression_techniques._sempipes_impl2 import sempipes_pipeline2
from sempipes.optimisers.trajectory import load_trajectory_from_json

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_log_error(y, y_predicted))


pipeline = sempipes_pipeline2()

trajectory = load_trajectory_from_json(
    ".experiments/househouse_prices_advanced_regression_techniques/colopro_20260102_064629_09325b55.json"
)
best_outcome = max(trajectory.outcomes, key=lambda x: (x.score, -x.search_node.trial))

data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
seed = 42
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_index, (train_idx, test_idx) in enumerate(kf.split(data)):
    data_train = data.iloc[train_idx]
    data_test = data.iloc[test_idx]
    np.random.seed(seed)
    random.seed(seed)

    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    env_fit = pipeline.skb.get_data()
    env_fit["data"] = data_train
    env_fit["sempipes_prefitted_state__house_features"] = best_outcome.state

    learner.fit(env_fit)

    env_eval = pipeline.skb.get_data()
    env_eval["data"] = data_test
    env_eval["sempipes_prefitted_state__house_features"] = best_outcome.state

    y_pred = learner.predict(env_eval)

    score = rmsle(data_test["SalePrice"], y_pred)
    print(f"RMSLE on split {fold_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
