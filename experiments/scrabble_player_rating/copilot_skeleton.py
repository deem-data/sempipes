# pylint: skip-file
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
games = pd.read_csv("experiments/scrabble_player_rating/games.csv")
nominal_features = ["time_control_name", "game_end_reason", "lexicon", "rating_mode"]
turns = pd.read_csv("experiments/scrabble_player_rating/turns.csv.gz")
all_data = pd.read_csv("experiments/scrabble_player_rating/data.csv")

all_players = all_data.nickname.unique()
non_bot_players = [player for player in all_players if player not in {"BetterBot", "HastyBot", "STEEBot"}]

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    train_players, test_players = train_test_split(non_bot_players, test_size=0.5, random_state=seed)
    train_game_ids = all_data[all_data.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all_data[all_data.nickname.isin(test_players)].game_id.unique()

    train = all_data[all_data.game_id.isin(train_game_ids)]
    test = all_data[all_data.game_id.isin(test_game_ids)]
    test_non_bot_mask = test.nickname.isin(test_players)

    # TODO add feature engineering code

    # TODO add model training and prediction

    y_pred = ...  # TODO Replace with actual predictions
    rmse = sqrt(mean_squared_error(test[test_non_bot_mask]["rating"], y_pred[test_non_bot_mask]))  # type: ignore[index]
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
