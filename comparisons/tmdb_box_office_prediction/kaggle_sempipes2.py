import random

import numpy as np
from sklearn.metrics import mean_squared_log_error

import sempipes
from comparisons.tmdb_box_office_prediction._sempipes_impl2 import sempipes_pipeline2

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="anthropic/claude-haiku-4-5-20251001",
        parameters={"temperature": 0.0},
    )
)

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    random.seed(seed)

    pipeline = sempipes_pipeline2("comparisons/tmdb_box_office_prediction/data.csv", seed)

    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    train_env = split["train"]
    test_env = split["test"]

    learner.fit(train_env)
    y_pred = learner.predict(test_env)

    rmsle = np.sqrt(mean_squared_log_error(test_env["_skrub_y"], y_pred))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
