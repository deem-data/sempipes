import random
import warnings

import numpy as np
from sklearn.metrics import mean_squared_log_error

import sempipes  # pylint: disable=unused-import
from experiments.house_prices_advanced_regression_techniques._sempipes_impl3 import sempipes_pipeline3

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_log_error(y, y_predicted))


pipeline = sempipes_pipeline3("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    random.seed(seed)
    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])
    score = rmsle(split["test"]["_skrub_y"], y_pred)
    print(f"RMSLE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
