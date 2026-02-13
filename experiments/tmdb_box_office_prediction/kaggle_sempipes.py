import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from experiments.tmdb_box_office_prediction._sempipes_impl import sempipes_pipeline

data = pd.read_csv("experiments/tmdb_box_office_prediction/data.csv")

pipeline = sempipes_pipeline()  # "experiments/tmdb_box_office_prediction/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    # split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    train, test = train_test_split(data, test_size=0.5, random_state=seed)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    env_train = pipeline.skb.get_data()
    env_train["data"] = train
    env_test = pipeline.skb.get_data()
    env_test["data"] = test

    learner.fit(env_train)
    y_pred = learner.predict(env_test)

    rmsle = np.sqrt(mean_squared_log_error(test["revenue"], y_pred))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
