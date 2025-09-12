from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
games = pd.read_csv("comparisons/scrabble_player_rating/games.csv")
turns = pd.read_csv("comparisons/scrabble_player_rating/turns.csv.gz")
all_data = pd.read_csv("comparisons/scrabble_player_rating/train.csv")

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

    # Merge the datasets on game_id
    merged_data = pd.merge(train, games, on="game_id")
    merged_data = pd.merge(
        merged_data,
        turns.groupby("game_id").agg({"points": "sum"}).reset_index(),
        on="game_id",
    )

    # Prepare the features and target variable
    X = merged_data[["game_duration_seconds", "winner", "points"]]
    y = merged_data["rating"]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Prepare the test set
    test_merged = pd.merge(test, games, on="game_id")
    test_merged = pd.merge(
        test_merged,
        turns.groupby("game_id").agg({"points": "sum"}).reset_index(),
        on="game_id",
    )
    X_test = test_merged[["game_duration_seconds", "winner", "points"]]

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_true = test[test_non_bot_mask]["rating"]
    rmse = sqrt(mean_squared_error(y_true, y_pred[test_non_bot_mask]))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
