import json
import random

import numpy as np
from sklearn.metrics import mean_squared_log_error

from experiments.tmdb_box_office_prediction._sempipes_impl2 import sempipes_pipeline2

with open("experiments/tmdb_box_office_prediction/_sempipes_state2.json", "r", encoding="utf-8") as f:
    state = json.load(f)

pipeline = sempipes_pipeline2("experiments/tmdb_box_office_prediction/data.csv", pipeline_seed=42)

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    random.seed(seed)

    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    env_fit = split["train"]
    env_fit["sempipes_prefitted_state__additional_movie_features"] = state

    learner.fit(env_fit)

    env_eval = split["test"]
    env_eval["sempipes_prefitted_state__additional_movie_features"] = state

    y_pred = learner.predict(env_eval)

    rmsle = np.sqrt(mean_squared_log_error(env_eval["_skrub_y"], y_pred))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
