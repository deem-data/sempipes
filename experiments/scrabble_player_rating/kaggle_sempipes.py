import numpy as np
from sklearn.metrics import mean_squared_error

import sempipes  # pylint: disable=unused-import
from experiments.scrabble_player_rating import custom_splitter
from experiments.scrabble_player_rating._sempipes_impl import sempipes_pipeline

pipeline = sempipes_pipeline("experiments/scrabble_player_rating/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5, splitter=custom_splitter)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])
    rmse = np.sqrt(mean_squared_error(split["test"]["_skrub_y"], y_pred))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
