from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
games = pd.read_csv("experiments/scrabble_player_rating/games.csv")
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

    ### MINI-SWE-AGENT ###
    # List of known bots
    bots = ["HastyBot", "BetterBot", "STEEBot"]

    # Filter out bot players
    df_humans = train[~train["nickname"].isin(bots)].copy()

    # Features and target
    X_train = df_humans[["score"]]  # Only 'score' is available as a feature
    y_train = df_humans["rating"]

    # Train a regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ###

    X_test = test[test_non_bot_mask][["score"]]
    y_test = test[test_non_bot_mask]["rating"]

    # Predict on the test set
    y_pred = model.predict(X_test)
    # y_true = test[test_non_bot_mask]["rating"]
    rmse = sqrt(mean_squared_error(y_test, y_pred))  # [test_non_bot_mask]))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
