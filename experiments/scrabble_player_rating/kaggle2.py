# Importing the Standard Libraries
import random
import warnings
from math import sqrt

import lightgbm as ltb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# from sklearn_pandas import DataFrameMapper
from scipy.special import boxcox, inv_boxcox
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

# Importing feature selection libraries
from sklearn.feature_selection import RFECV, SelectFromModel

# Importing the libraries to train the model
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV

# Importing the metrics libraries
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

# importing libraries to process data
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from experiments.scrabble_player_rating import BOT_NAMES

warnings.filterwarnings("ignore")

# Load the data
df_games = pd.read_csv("experiments/scrabble_player_rating/games.csv")
# nominal_features = ["time_control_name", "game_end_reason", "lexicon", "rating_mode"]
df_turns = pd.read_csv("experiments/scrabble_player_rating/turns.csv.gz")
all_data = pd.read_csv("experiments/scrabble_player_rating/data.csv")

all_players = all_data.nickname.unique()
non_bot_players = [player for player in all_players if player not in BOT_NAMES]

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    random.seed(seed)
    train_players, test_players = train_test_split(non_bot_players, test_size=0.5, random_state=seed)
    train_game_ids = all_data[all_data.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all_data[all_data.nickname.isin(test_players)].game_id.unique()

    df_train = all_data[all_data.game_id.isin(train_game_ids)]
    df_test = all_data[all_data.game_id.isin(test_game_ids)]

    df_test_non_bot_mask = df_test.nickname.isin(test_players)
    y_test = df_test[df_test_non_bot_mask].copy(deep=True)["rating"]

    # Merging df_train with df_games
    df_train = df_train.merge(df_games, how="inner", on="game_id")
    df_test = df_test.merge(df_games, how="inner", on="game_id")

    # adding a column with the len of the move
    def transform_move(tamanho):
        return len(str(tamanho))

    df_turns["len_move"] = df_turns["move"].map(transform_move)
    # adding aggregation with turn_number', 'len_move'
    df_turns_agg1 = df_turns.groupby(["game_id", "nickname"], as_index=False).agg(
        {"turn_number": "count", "len_move": "mean"}
    )
    # adding aggregation with 'points':['mean', 'min', 'max']
    df_turns_agg2 = df_turns.groupby(["game_id", "nickname"], as_index=False).agg({"points": ["mean", "min", "max"]})
    df_turns_agg2 = df_turns_agg2["points"].rename(
        columns={"mean": "points_mean", "min": "points_min", "max": "points_max"}
    )
    # Merging the aggregations
    df_turns_agg = pd.concat([df_turns_agg1, df_turns_agg2], axis=1)

    # Merging df_train with aggregations
    df_train = df_train.merge(df_turns_agg, how="inner", on=["game_id", "nickname"])
    df_test = df_test.merge(df_turns_agg, how="inner", on=["game_id", "nickname"])

    X = df_train.drop(["game_id"], axis=1)
    # adding a column with the opponent's rating

    # X['other_rating'] = 0
    # for i in range(0, len(X), 2):
    #   if pd.isna(X['rating'][i]) == True:
    #     X['other_rating'][i]= X['rating'][i+1]
    #     X['other_rating'][i+1]= X['rating'][i+1]
    #   elif pd.isna(X['rating'][i+1]) == True:
    #     X['other_rating'][i+1]= X['rating'][i]
    #     X['other_rating'][i]= X['rating'][i]
    #   else:
    #     X['other_rating'][i] = X['rating'][i+1]
    #     X['other_rating'][i+1] = X['rating'][i]
    #
    # df_test['other_rating'] = 0
    # for i in range(0, len(df_test), 2):
    #   if pd.isna(df_test['rating'][i]) == True:
    #     df_test['other_rating'][i]= df_test['rating'][i+1]
    #     df_test['other_rating'][i+1]= df_test['rating'][i+1]
    #   elif pd.isna(df_test['rating'][i+1]) == True:
    #     df_test['other_rating'][i+1]= df_test['rating'][i]
    #     df_test['other_rating'][i]= df_test['rating'][i]
    #   else:
    #     df_test['other_rating'][i] = df_test['rating'][i+1]
    #     df_test['other_rating'][i+1] = df_test['rating'][i]

    # adding a column with the player's average points
    df_adiciona2 = X.groupby(["nickname"], as_index=False).agg({"score": "mean"})
    df_adiciona2 = df_adiciona2.rename(columns={"score": "media_nickname"})
    X = X.merge(df_adiciona2, how="inner", on="nickname")

    df_adiciona3 = df_test.groupby(["nickname"], as_index=False).agg({"score": "mean"})
    df_adiciona3 = df_adiciona3.rename(columns={"score": "media_nickname"})
    df_test = df_test.merge(df_adiciona3, how="inner", on="nickname")
    # adding a column of 1 if the player wins and 0 if he loses
    # adding a column to the player and opponent score difference
    # adding a column with the opponent's average points
    X["Win"] = 0
    X["score_diff"] = 0
    X["other_media_nickname"] = 0
    for i in range(0, len(X), 2):
        X["score_diff"][i] = X["score"][i] - X["score"][i + 1]
        X["score_diff"][i + 1] = X["score"][i + 1] - X["score"][i]

        X["other_media_nickname"][i] = X["media_nickname"][i + 1]
        X["other_media_nickname"][i + 1] = X["media_nickname"][i]
        if X["score_diff"][i] > 0:
            X["Win"][i] = 1
        else:
            X["Win"][i + 1] = 1

    df_test["Win"] = 0
    df_test["score_diff"] = 0
    df_test["other_media_nickname"] = 0
    for i in range(0, len(df_test), 2):
        df_test["score_diff"][i] = df_test["score"][i] - df_test["score"][i + 1]
        df_test["score_diff"][i + 1] = df_test["score"][i + 1] - df_test["score"][i]

        df_test["other_media_nickname"][i] = df_test["media_nickname"][i + 1]
        df_test["other_media_nickname"][i + 1] = df_test["media_nickname"][i]

        if df_test["score_diff"][i] > 0:
            df_test["Win"][i] = 1
        else:
            df_test["Win"][i + 1] = 1
    # adding a column with the average points difference between the player and the opponent
    X["diff_media_nickname"] = X["media_nickname"] - X["other_media_nickname"]
    df_test["diff_media_nickname"] = df_test["media_nickname"] - df_test["other_media_nickname"]

    # turning time_control_name column to regular or not
    def transform_time_control_name(name):
        if name == "regular":
            return 0
        else:
            return 1

    X["time_control_name_num"] = X["time_control_name"].map(transform_time_control_name)
    df_test["time_control_name_num"] = df_test["time_control_name"].map(transform_time_control_name)

    # turning the game_end_reason column into standard or not
    def transform_game_end_reason(end_reason):
        if end_reason == "STANDARD":
            return 0
        else:
            return 1

    X["game_end_reason_num"] = X["game_end_reason"].map(transform_game_end_reason)
    df_test["game_end_reason_num"] = df_test["game_end_reason"].map(transform_game_end_reason)

    # turning rating_mode column to RATED or not
    def transform_rating_mode(mode):
        if mode == "RATED":
            return 1
        else:
            return 0

    X["rating_mode_num"] = X["rating_mode"].map(transform_rating_mode)
    df_test["rating_mode_num"] = df_test["rating_mode"].map(transform_rating_mode)

    # turning the lexicon column into a numeric variable
    def transform_lexicon(lex):
        if lex == "CSW21":
            return 2
        elif lex == "NWL20":
            return 1
        else:
            return 0

    X["lexicon_num"] = X["lexicon"].map(transform_lexicon)
    df_test["lexicon_num"] = df_test["lexicon"].map(transform_lexicon)
    # adding a column of 1 if the player started the match and 0 if the opponent started
    X["first_num"] = 0
    for i in range(0, len(X), 2):
        if X["first"][i] == X["nickname"][i]:
            X["first_num"][i] = 1
            X["first_num"][i + 1] = 0
        else:
            X["first_num"][i] = 0
            X["first_num"][i + 1] = 1

    df_test["first_num"] = 0
    # print('###', len(df_test), len(X))
    for i in range(0, len(df_test), 2):
        # if df_test['first'][i] == X['nickname'][i]:
        if df_test["first"][i] == df_test["nickname"][i]:
            df_test["first_num"][i] = 1
            df_test["first_num"][i + 1] = 0
        else:
            df_test["first_num"][i] = 0
            df_test["first_num"][i + 1] = 1
    # adding a column with the initial_time_seconds and the duration of the game
    X["time_difference"] = X["initial_time_seconds"] - X["game_duration_seconds"]
    df_test["time_difference"] = df_test["initial_time_seconds"] - df_test["game_duration_seconds"]
    # adding a column the average time spent on each turn
    X["time_per_turn"] = X["game_duration_seconds"] / X["turn_number"]
    df_test["time_per_turn"] = df_test["game_duration_seconds"] / df_test["turn_number"]
    # adding a column with the average points spent per game time
    X["points_per_second"] = X["points_mean"] / X["game_duration_seconds"]
    df_test["points_per_second"] = df_test["points_mean"] / df_test["game_duration_seconds"]

    X = X[~X.nickname.str.endswith("Bot")]
    df_test = df_test[~df_test.nickname.str.endswith("Bot")]

    y = stats.boxcox(X["rating"], -1.4)

    # choosing some variables
    Xvariaveis = [
        "score",
        "Win",
        "game_duration_seconds",
        "initial_time_seconds",
        "rating_mode_num",
        "max_overtime_minutes",
        "first_num",
        "time_control_name_num",
        "lexicon_num",
        "game_end_reason_num",
        "time_difference",  #'other_rating',
        "points_mean",
        "turn_number",
        "time_per_turn",
        "len_move",
        "score_diff",
        "points_per_second",
        "media_nickname",
        "other_media_nickname",
        "diff_media_nickname",
    ]

    # print([c for c in X.columns if c not in Xvariaveis])

    # splitting test data
    X = X[Xvariaveis]
    X_test = df_test[Xvariaveis]
    y_test = df_test["rating"]
    # splitting the data into training and validation
    # Xtrain, Xval, ytrain, yval = train_test_split(X, y, train_size=0.7, random_state=0, shuffle=False)

    # normalizing the data

    # mapper = DataFrameMapper([(X.columns, StandardScaler())])
    # scaled_features = mapper.fit_transform(X.copy())
    # X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    #
    # mapper = DataFrameMapper([(X_test.columns, StandardScaler())])
    # scaled_features = mapper.fit_transform(X_test.copy())
    # X_test = pd.DataFrame(scaled_features, index=X_test.index, columns=X_test.columns)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # creating the final model
    modelo1 = ExtraTreesRegressor(random_state=42)
    modelo1.fit(X, y)
    p_extreg = inv_boxcox(modelo1.predict(X_test.fillna(-1)), -1.4)

    modelo2 = ltb.LGBMRegressor(
        learning_rate=0.08797697229790209,
        num_leaves=12,
        min_child_samples=2,
        random_state=42,
        subsample=0.29682698086764914,
        colsample_bytree=0.2789636201075709,
        verbosity=-1,
    )
    modelo2.fit(X, y)
    p_lgbm = inv_boxcox(modelo2.predict(X_test.fillna(-1)), -1.4)

    modelo3 = RandomForestRegressor(random_state=42)
    modelo3.fit(X, y)
    p_ranforeg = inv_boxcox(modelo3.predict(X_test.fillna(-1)), -1.4)

    modelo4 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=800, max_depth=6, random_state=42)
    modelo4.fit(X, y)
    p_graboor = inv_boxcox(modelo4.predict(X_test.fillna(-1)), -1.4)

    p_ensemble = (p_extreg + p_lgbm + p_ranforeg + p_graboor) / 4
    print("-->", len(y_test))
    rmse = sqrt(mean_squared_error(y_test, p_ensemble))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
