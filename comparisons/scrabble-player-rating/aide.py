import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data
games = pd.read_csv("comparisons/scrabble-player-rating/games.csv")
turns = pd.read_csv("comparisons/scrabble-player-rating/turns.csv")
all = pd.read_csv("comparisons/scrabble-player-rating/train.csv")

all_players = all.nickname.unique()
non_bot_players = [player for player in all_players if player not in {'BetterBot', 'HastyBot', 'STEEBot'}]

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):

    train_players, test_players = train_test_split(non_bot_players, test_size=0.5, random_state=seed)
    train_game_ids = all[all.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all[all.nickname.isin(test_players)].game_id.unique()

    train = all[all.game_id.isin(train_game_ids)]
    test = all[all.game_id.isin(test_game_ids)]
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

    print("TRAIN SPLIT", len(merged_data), np.sum(merged_data["game_id"]))

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
    print("TEST SPLIT", len(y_true), np.sum(test[test_non_bot_mask]["game_id"]))
    rmse = sqrt(mean_squared_error(y_true, y_pred[test_non_bot_mask]))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
