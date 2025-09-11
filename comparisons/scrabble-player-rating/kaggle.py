import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from math import sqrt

# Load the data
games = pd.read_csv("comparisons/scrabble-player-rating/games.csv")
nominal_features = ["time_control_name", "game_end_reason", "lexicon", "rating_mode"]
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

    def sum_first_five(series):
        return sum(series.values[::-1][:5])


    def replace_winner(row, data):
        """Set the value of winner to 1 if the player won, -1 if the lost, or 0 if it was a draw."""
        # Locate opponent as the row with the same game_id but different nickname.
        opponent_row = data.loc[(data.game_id == row.loc["game_id"]) & (data.nickname != row.loc["nickname"])]

        # Compare scores. Set the winner to 1, the loser to -1 and if a tie, give both 0.
        if (row.loc["score"] > opponent_row["score"].values).all():
            row.loc["winner"] = 1
        elif (row.loc["score"] < opponent_row["score"].values).all():
            row.loc["winner"] = -1
        else:
            row.loc["winner"] = 0
        return row


    def replace_first(row):
        """Set the value in column first to 1 if the player went first in their game,
        or to 0 if they went second."""
        if row.loc["first"] == row.loc["nickname"]:
            row.loc["first"] = 1
        else:
            row.loc["first"] = 0
        return row


    def relabel_values(data):
        def relabel(row):
            row = replace_winner(row, data)
            row = replace_first(row)
            return row
        return relabel


    #nominal_features = ["time_control_name", "game_end_reason", "lexicon", "rating_mode"]
    non_informative_features = ["created_at", "nickname", "game_id"]

    total_turns = turns.groupby(["game_id", "nickname"]).turn_number.count()
    max_points = turns.groupby(["game_id", "nickname"]).points.max()
    min_points = turns.groupby(["game_id", "nickname"]).points.min()
    first_five_turn_point_sum = turns.groupby(["game_id", "nickname"]).points.agg(sum_first_five)

    game_player_data = total_turns.reset_index()
    game_player_data.rename(columns={"turn_number": "total_turns"}, inplace=True)
    game_player_data["first_five_turns_points"] = first_five_turn_point_sum.reset_index()["points"]
    game_player_data["max_points_turn"] = max_points.reset_index()["points"]
    game_player_data["min_points_turn"] = min_points.reset_index()["points"]
    game_player_data["max_min_difference"] = game_player_data.max_points_turn - game_player_data.min_points_turn
    game_player_data = game_player_data.join(games.set_index("game_id"), how="left", on="game_id")
    game_player_data["time_used"] = game_player_data.game_duration_seconds / game_player_data.initial_time_seconds


    train_data = pd.merge(train, game_player_data, how="left", left_on=["game_id", "nickname"], right_on=["game_id", "nickname"])
    train_data["points_per_turn"] = train_data.score / train_data.total_turns
    train_data["points_per_second"] = train_data.score / train_data.game_duration_seconds
    train_data["time_used"] = train_data.game_duration_seconds / train_data.initial_time_seconds

    y = train_data["rating"]
    X = train_data.apply(relabel_values(train_data), axis=1)

    one_hot_encoded_features = pd.get_dummies(X[nominal_features])
    X_encoded = pd.concat((X, one_hot_encoded_features), axis=1)
    print("TRAIN SPLIT", len(X_encoded), np.sum(X_encoded["game_id"]))
    X_encoded.drop(columns=nominal_features + non_informative_features, axis=1, inplace=True)
    X_encoded = X_encoded.drop(columns=["rating"])



    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, random_state=42)

    xgb_reg = XGBRegressor(
        colsample_bytree=0.4,
        gamma=0,
        learning_rate=0.07,
        max_depth=3,
        min_child_weight=1.5,
        n_estimators=8000,
        reg_alpha=0.75,
        reg_lambda=0.45,
        subsample=0.6,
        seed=42,
        # tree_method='gpu_hist',
        eval_metric="rmse",
        early_stopping_rounds=1000
    )

    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    test_data = pd.merge(test, game_player_data, how="left", left_on=["game_id", "nickname"], right_on=["game_id", "nickname"])
    test_data["points_per_turn"] = test_data.score / test_data.total_turns
    test_data["points_per_second"] = test_data.score / test_data.game_duration_seconds

    X_test = test_data.apply(relabel_values(test_data), axis=1)
    X_test["time_used"] = X_test.game_duration_seconds / X_test.initial_time_seconds
    X_test['lexicon'] = pd.Categorical(X_test['lexicon'], categories=['CSW21', 'ECWL', 'NSWL20', 'NWL20'])
    one_hot_encoded_features = pd.get_dummies(X_test[nominal_features])
    X_test_encoded = pd.concat((X_test, one_hot_encoded_features), axis=1)
    X_test_encoded.drop(columns=nominal_features + non_informative_features, axis=1, inplace=True)

    y_true = X_test_encoded["rating"]
    X_test_encoded = X_test_encoded.drop(columns=["rating"])

    if not 'lexicon_NSWL20' in list(X_test_encoded.columns):
        X_test_encoded['lexicon_NSWL20'] = 0.0

    y_pred = xgb_reg.predict(X_test_encoded)
    rmse = sqrt(mean_squared_error(test[test_non_bot_mask]["rating"], y_pred[test_non_bot_mask]))
    print("TEST SPLIT", len(test[test_non_bot_mask]), np.sum(test[test_non_bot_mask]["game_id"]))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
