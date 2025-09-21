import numpy as np
import pandas as pd
import skrub
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

games_df = pd.read_csv("comparisons/scrabble_player_rating/games.csv")
turns_df = pd.read_csv("comparisons/scrabble_player_rating/turns.csv.gz")
data_df = pd.read_csv("comparisons/scrabble_player_rating/data.csv")

games = skrub.var("games", games_df)
turns = skrub.var("turns", turns_df)
data = skrub.var("data", data_df).skb.mark_as_X()


def sum_first_five(series):
    return sum(series.values[::-1][:5])


def replace_winner(row, data_split):
    """Set the value of winner to 1 if the player won, -1 if the lost, or 0 if it was a draw."""
    # Locate opponent as the row with the same game_id but different nickname.
    opponent_row = data_split.loc[
        (data_split.game_id == row.loc["game_id"]) & (data_split.nickname != row.loc["nickname"])
    ]

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


def relabel_values(data_split):
    def relabel(row):
        row = replace_winner(row, data_split)
        row = replace_first(row)
        return row

    return relabel


total_turns = turns.groupby(["game_id", "nickname"]).turn_number.count()
max_points = turns.groupby(["game_id", "nickname"]).points.max()
min_points = turns.groupby(["game_id", "nickname"]).points.min()
first_five_turn_point_sum = turns.groupby(["game_id", "nickname"]).points.agg(sum_first_five)

game_player_data = total_turns.reset_index()
game_player_data = game_player_data.rename(columns={"turn_number": "total_turns"})

game_player_data = game_player_data.assign(
    first_five_turns_points=first_five_turn_point_sum.reset_index()["points"],
    max_points_turn=max_points.reset_index()["points"],
    min_points_turn=min_points.reset_index()["points"],
)
game_player_data = game_player_data.assign(
    max_min_difference=game_player_data.max_points_turn - game_player_data.min_points_turn
)

game_player_data = game_player_data.join(games.set_index("game_id"), how="left", on="game_id")
game_player_data = game_player_data.assign(
    time_used=game_player_data.game_duration_seconds / game_player_data.initial_time_seconds
)

data = data.merge(game_player_data, how="left", left_on=["game_id", "nickname"], right_on=["game_id", "nickname"])
data = data.assign(
    points_per_turn=data.score / data.total_turns,
    points_per_second=data.score / data.game_duration_seconds,
    time_used=data.game_duration_seconds / data.initial_time_seconds,
)

X = data.skb.apply_func(lambda df: df.apply(relabel_values(df), axis=1))
y = X["rating"].skb.mark_as_y()
X = X.drop(columns=["rating"])


pass_through_features = [
    "score",
    "total_turns",
    "first_five_turns_points",
    "max_points_turn",
    "min_points_turn",
    "max_min_difference",
    "first",
    "winner",
    "initial_time_seconds",
    "increment_seconds",
    "max_overtime_minutes",
    "game_duration_seconds",
    "time_used",
    "points_per_turn",
    "points_per_second",
]
nominal_features = ["time_control_name", "game_end_reason", "rating_mode"]

encoder = ColumnTransformer(
    transformers=[
        ("ordinal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
        (
            "ordinal_lexicon",
            OneHotEncoder(categories=[["NWL20", "CSW21", "ECWL", "NSWL20"]], sparse_output=False),
            ["lexicon"],
        ),
        ("passthrough", "passthrough", pass_through_features),
    ]
)

X = X.skb.apply(encoder)

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
    # early_stopping_rounds=1000
)

predictions = X.skb.apply(xgb_reg, y=y)


def custom_splitter(all_data, y, random_state, test_size):  # pylint: disable=unused-argument,redefined-outer-name
    all_players = all_data.nickname.unique()
    non_bot_players = [player for player in all_players if player not in {"BetterBot", "HastyBot", "STEEBot"}]
    train_players, test_players = train_test_split(non_bot_players, test_size=test_size, random_state=random_state)
    train_game_ids = all_data[all_data.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all_data[all_data.nickname.isin(test_players)].game_id.unique()

    train = all_data[all_data.game_id.isin(train_game_ids)]
    test = all_data[all_data.game_id.isin(test_game_ids)]
    test = test[test.nickname.isin(test_players)]
    print("TRAIN SPLIT", len(train), np.sum(train["game_id"]))
    print("TEST SPLIT", len(test), np.sum(test["game_id"]))

    return train, test, train.rating, test.rating


scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = predictions.skb.train_test_split(random_state=seed, test_size=0.5, splitter=custom_splitter)
    learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])
    rmse = np.sqrt(mean_squared_error(split["test"]["_skrub_y"], y_pred))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
