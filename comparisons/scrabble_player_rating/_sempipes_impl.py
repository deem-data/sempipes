import numpy as np
import pandas as pd
import skrub
from sklearn.preprocessing import OneHotEncoder
from skrub import selectors as s
from xgboost import XGBRegressor

import sempipes  # pylint: disable=unused-import


def sempipes_pipeline(data_file):
    games_df = pd.read_csv("comparisons/scrabble_player_rating/games.csv")
    turns_df = pd.read_csv("comparisons/scrabble_player_rating/turns.csv.gz")
    data_df = pd.read_csv(data_file)

    games = skrub.var("games", games_df)
    turns = skrub.var("turns", turns_df)
    data = skrub.var("data", data_df).skb.mark_as_X()

    def sum_first_five(series):
        return sum(series.values[::-1][:5])

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
    data = data.merge(game_player_data, how="left", left_on=["game_id", "nickname"], right_on=["game_id", "nickname"])

    X = data
    y = X["rating"].skb.mark_as_y()
    X = X.drop(columns=["rating"])

    X = X.with_sem_features(
        nl_prompt="""
        Create additional features that could help predict the rating of a player. Such features could relate to how
        the player scores when they go first in a round and to which games they won, how often they won games, etc.
    
        Furthermore, consider how much time a player typically uses, how many points per turn they make, how many points per second, etc.
    
        """,
        name="player_features",
        how_many=15,
    )

    X_encoded = X.skb.apply(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        cols=(s.categorical() - s.cols("lexicon")),
    )
    X_encoded = X_encoded.skb.apply(
        OneHotEncoder(categories=[["NWL20", "CSW21", "ECWL", "NSWL20"]], sparse_output=False),
        cols=s.cols("lexicon"),
    )

    X_encoded = X_encoded.skb.drop(["created_at", "nickname", "game_id"])
    X_encoded = X_encoded.skb.select(s.numeric() | s.boolean())

    X_encoded = X_encoded.skb.apply_func(lambda x: x.replace([np.inf, -np.inf], np.nan))

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

    predictions = X_encoded.skb.apply(xgb_reg, y=y)
    return predictions
