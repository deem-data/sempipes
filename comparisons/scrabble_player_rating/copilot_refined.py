# pylint: skip-file
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the data
games = pd.read_csv("comparisons/scrabble_player_rating/games.csv")
nominal_features = ["time_control_name", "game_end_reason", "lexicon", "rating_mode"]
turns = pd.read_csv("comparisons/scrabble_player_rating/turns.csv.gz")
all_data = pd.read_csv("comparisons/scrabble_player_rating/data.csv")

all_players = all_data.nickname.unique()
non_bot_players = [player for player in all_players if player not in {"BetterBot", "HastyBot", "STEEBot"}]

# --- Feature Engineering ---
# Aggregate turn features per player per game
agg_dict = {"turn_number": "count"}
if "is_bingo" in turns.columns:
    agg_dict["is_bingo"] = "sum"

turn_agg = turns.groupby(["game_id", "nickname"]).agg(agg_dict).reset_index()
turn_agg = turn_agg.rename(columns={"turn_number": "num_turns", "is_bingo": "bingo_count"})

# Merge game features
data_merged = all_data.merge(games, on="game_id", how="left")
data_merged = data_merged.merge(turn_agg, on=["game_id", "nickname"], how="left")

# Fill missing turn features with 0
for col in ["num_turns", "bingo_count"]:
    if col in data_merged.columns:
        data_merged[col] = data_merged[col].fillna(0)

# Encode nominal features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_nominals = encoder.fit_transform(data_merged[nominal_features])
encoded_nominals_df = pd.DataFrame(encoded_nominals, columns=encoder.get_feature_names_out(nominal_features))
data_merged = pd.concat([data_merged.reset_index(drop=True), encoded_nominals_df], axis=1)
# Select features for modeling
feature_cols = [
    "score",
    "num_turns",
    "bingo_count",
    "initial_time_seconds",
    "increment_seconds",
    "max_overtime_minutes",
    "game_duration_seconds",
] + list(encoded_nominals_df.columns)
feature_cols = [col for col in feature_cols if col in data_merged.columns]

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    train_players, test_players = train_test_split(non_bot_players, test_size=0.5, random_state=seed)
    train_game_ids = all_data[all_data.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all_data[all_data.nickname.isin(test_players)].game_id.unique()

    train = data_merged[data_merged.game_id.isin(train_game_ids)]
    test = data_merged[data_merged.game_id.isin(test_game_ids)]
    test_non_bot_mask = test.nickname.isin(test_players)

    # --- Model Training and Prediction ---
    X_train = train[feature_cols]
    y_train = train["rating"]
    X_test = test[feature_cols]
    y_test = test["rating"]

    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test[test_non_bot_mask], y_pred[test_non_bot_mask]))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
