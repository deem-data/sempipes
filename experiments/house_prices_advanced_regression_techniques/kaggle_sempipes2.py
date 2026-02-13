import random
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold, train_test_split

import sempipes  # pylint: disable=unused-import
from experiments.house_prices_advanced_regression_techniques._sempipes_impl2 import sempipes_pipeline2

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_log_error(y, y_predicted))


pipeline = sempipes_pipeline2()

data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
seed = 42
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_index, (train_idx, test_idx) in enumerate(kf.split(data)):
    np.random.seed(seed)
    random.seed(seed)
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    # train, test = train_test_split(data, test_size=0.5, random_state=seed)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    env_train = pipeline.skb.get_data()
    env_train["data"] = train
    env_test = pipeline.skb.get_data()
    env_test["data"] = test
    learner.fit(env_train)
    y_pred = learner.predict(env_test)
    score = rmsle(test["SalePrice"], y_pred)
    print(f"RMSLE on {fold_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
