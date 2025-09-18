import numpy as np
from sklearn.metrics import mean_squared_log_error

from comparisons.tmdb_box_office_prediction._gyyre_impl import gyyre_pipeline

pipeline = gyyre_pipeline("comparisons/tmdb_box_office_prediction/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])

    rmsle = np.sqrt(mean_squared_log_error(split["test"]["_skrub_y"], y_pred))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
