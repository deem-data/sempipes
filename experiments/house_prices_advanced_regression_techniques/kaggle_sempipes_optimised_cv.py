import json
import warnings
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import numpy as np

import sempipes  # pylint: disable=unused-import
from experiments.house_prices_advanced_regression_techniques._sempipes_impl import rmsle, sempipes_pipeline
from sempipes.optimisers.trajectory import load_trajectory_from_json

warnings.filterwarnings("ignore")

pipeline = sempipes_pipeline()

trajectory = load_trajectory_from_json(".experiments/house_prices_advanced_regression_techniques/colopro_20260103_082657_6ff65fb9.json")
best_outcome = max(trajectory.outcomes, key=lambda x: (x.score, -x.search_node.trial))
state = best_outcome.state

data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_index, (train_idx, test_idx) in enumerate(kf.split(data)):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    env_fit = pipeline.skb.get_data()
    env_fit["data"] = train
    env_fit["sempipes_prefitted_state__house_features"] = state

    learner.fit(env_fit)

    env_eval = pipeline.skb.get_data()
    env_eval["data"] = test
    env_eval["sempipes_prefitted_state__house_features"] = state

    y_pred = learner.predict(env_eval)

    score = rmsle(np.log1p(test["SalePrice"]), y_pred)
    print(f"RMSLE on split {fold_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
