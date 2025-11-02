import warnings

import lightgbm as lgb
import pandas as pd
import skrub
from scipy.special import boxcox, inv_boxcox  # pylint: disable=no-name-in-module
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model1 = ExtraTreesRegressor(random_state=42)
        self.model2 = lgb.LGBMRegressor(
            learning_rate=0.08797697229790209,
            num_leaves=12,
            min_child_samples=2,
            subsample=0.29682698086764914,
            colsample_bytree=0.2789636201075709,
            verbosity=-1,
            random_state=42,
        )
        self.model3 = RandomForestRegressor(random_state=42)
        self.model4 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=800, max_depth=6, random_state=42)

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        self.model3.fit(X, y)
        self.model4.fit(X, y)
        return self

    def predict(self, X):
        p_extreg = inv_boxcox(self.model1.predict(X), -1.4)
        p_lgbm = inv_boxcox(self.model2.predict(X), -1.4)
        p_ranforeg = inv_boxcox(self.model3.predict(X), -1.4)
        p_graboor = inv_boxcox(self.model4.predict(X), -1.4)
        p_ensemble = (p_extreg + p_lgbm + p_ranforeg + p_graboor) / 4.0
        return p_ensemble


def sempipes_pipeline2(data_file):
    games_df = pd.read_csv("experiments/scrabble_player_rating/games.csv")
    turns_df = pd.read_csv("experiments/scrabble_player_rating/turns.csv.gz")
    data_df = pd.read_csv(data_file)

    games = skrub.var("games", games_df)
    df_turns = skrub.var("turns", turns_df)
    data = skrub.var("data", data_df).skb.subsample(n=100)

    data = data.merge(games, on="game_id", how="inner")

    # adding a column with the len of the move
    def transform_move(tamanho):
        return len(str(tamanho))

    df_turns = df_turns.assign(len_move=df_turns["move"].map(transform_move))
    df_turns_agg1 = df_turns.groupby(["game_id", "nickname"], as_index=False).agg(
        {"turn_number": "count", "len_move": "mean"}
    )
    df_turns_agg2 = df_turns.groupby(["game_id", "nickname"], as_index=False).agg({"points": ["mean", "min", "max"]})
    df_turns_agg2 = df_turns_agg2["points"].rename(
        columns={"mean": "points_mean", "min": "points_min", "max": "points_max"}
    )
    df_turns_agg = df_turns_agg1.skb.concat([df_turns_agg2])

    data = data.merge(df_turns_agg, how="inner", on=["game_id", "nickname"])

    def add_other(X):
        # adding a column with the player's average points
        df_adiciona2 = X.groupby(["nickname"], as_index=False).agg({"score": "mean"})
        df_adiciona2 = df_adiciona2.rename(columns={"score": "mean_score"})
        X = X.merge(df_adiciona2, how="inner", on="nickname")

        X["Win"] = 0
        X["score_diff"] = 0
        X["other_mean_score"] = 0.0
        for i in range(0, len(X), 2):
            X["score_diff"][i] = X["score"][i] - X["score"][i + 1]
            X["score_diff"][i + 1] = X["score"][i + 1] - X["score"][i]

            X["other_mean_score"][i] = X["mean_score"][i + 1]
            X["other_mean_score"][i + 1] = X["mean_score"][i]
            if X["score_diff"][i] > 0:
                X["Win"][i] = 1
            else:
                X["Win"][i + 1] = 1

        # adding a column with the average points difference between the player and the opponent
        X["diff_mean_score"] = X["mean_score"] - X["other_mean_score"]

        # # adding a column of 1 if the player started the match and 0 if the opponent started
        X["first_num"] = 0
        for i in range(0, len(X), 2):
            if X["first"][i] == X["nickname"][i]:
                X["first_num"][i] = 1
                X["first_num"][i + 1] = 0
            else:
                X["first_num"][i] = 0
                X["first_num"][i + 1] = 1

        # adding a column with the initial_time_seconds and the duration of the game
        X["time_difference"] = X["initial_time_seconds"] - X["game_duration_seconds"]

        return X

    data = data.skb.apply_func(add_other)
    data = data[~data.nickname.str.endswith("Bot")]

    X_d = data.skb.mark_as_X()
    y = X_d["rating"].skb.mark_as_y().skb.apply_func(lambda y: boxcox(y, -1.4))
    X_d = X_d.drop(columns=["rating"])

    Xvariaveis = [
        "score",
        "Win",
        "game_duration_seconds",
        "initial_time_seconds",
        "max_overtime_minutes",
        "first_num",
        "time_difference",
        "points_mean",
        "turn_number",
        "len_move",
        "score_diff",
        "mean_score",
        "other_mean_score",
        "diff_mean_score",
        "lexicon",
        "rating_mode",
        "time_control_name",
        "game_end_reason",
    ]
    X_d = X_d[Xvariaveis]

    X_d = X_d.sem_gen_features(
        nl_prompt="""
        Create additional features that could help predict the rating of a player. The two players for a game can be found by checking the rows with the same game_id.
        
        Definitely include the following:
         - the average time spent on each turn
         - the average points per second
         - a binary feature indicating whether the rating_mode is RATED or not
         - a binary feature indicating whether the game_end_reason is STANDARD or not
         - a binary feature indicating whether the time_control_name is regular or not
        
        """,
        name="player_features",
        how_many=15,
    )

    X_d = X_d.skb.apply(skrub.TableVectorizer())
    X_d = X_d.fillna(-1)
    X_d = X_d.replace([float("inf"), -float("inf")], -1)

    predictions = X_d.skb.apply(EnsembleRegressor(), y=y)
    return predictions
