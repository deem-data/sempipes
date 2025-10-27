import json
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import sempipes  # pylint: disable=unused-import
from experiments.scrabble_player_rating import BOT_NAMES
from experiments.scrabble_player_rating._sempipes_impl import sempipes_pipeline

warnings.filterwarnings("ignore")

with open("experiments/scrabble_player_rating/_sempipes_state.json", "r", encoding="utf-8") as f:
    state = json.load(f)


pipeline = sempipes_pipeline("experiments/scrabble_player_rating/data.csv")


def custom_splitter(all_data, y, random_state, test_size):  # pylint: disable=unused-argument
    all_players = all_data.nickname.unique()
    non_bot_players = [player for player in all_players if player not in BOT_NAMES]
    train_players, test_players = train_test_split(non_bot_players, test_size=test_size, random_state=random_state)
    train_game_ids = all_data[all_data.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all_data[all_data.nickname.isin(test_players)].game_id.unique()

    train = all_data[all_data.game_id.isin(train_game_ids)]
    test = all_data[all_data.game_id.isin(test_game_ids)]
    test = test[test.nickname.isin(test_players)]

    return train, test, train.rating, test.rating


scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5, splitter=custom_splitter)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    env_fit = split["train"]
    env_fit["sempipes_prefitted_state__player_features"] = state

    learner.fit(env_fit)

    env_eval = split["test"]
    env_eval["sempipes_prefitted_state__player_features"] = state

    y_pred = learner.predict(env_eval)

    rmse = np.sqrt(mean_squared_error(split["test"]["_skrub_y"], y_pred))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
