from math import sqrt

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import sempipes
from sempipes.optimisers.trajectory import load_trajectory_from_json

trajectory = load_trajectory_from_json(".experiments/scrabble_player_rating/colopro_20251224_070218_ce3f04e1.json")
best_outcome = max(trajectory.outcomes, key=lambda x: (x.score, -x.search_node.trial))


def sempipes_pipeline():
    data = skrub.var("data").skb.mark_as_X()
    games = skrub.var("games")

    merged_data = data.join(games, on="game_id", rsuffix="_games")

    y_bot_mask = merged_data["nickname"].isin(["HastyBot", "BetterBot", "STEEBot"])

    y = merged_data["rating"].skb.mark_as_y()
    X = merged_data.drop(columns=["rating"])

    X_with_features = X.sem_gen_features(
        nl_prompt="""
        Create additional features that could help predict the rating of a player. Such features could relate to how
        the player scores when they go first in a round and to which games they won, how often they won games, etc.
    
        Furthermore, consider how much time a player typically uses, how many points per turn they make, how many points per second, etc.
    
        """,
        name="player_features",
        how_many=10,
    )

    X_final = X_with_features[~X_with_features["nickname"].isin(["HastyBot", "BetterBot", "STEEBot"])]
    y = y[~y_bot_mask]

    X_final = X_final.skb.apply_func(lambda x: x.replace([float("inf"), -float("inf")], 0))
    X_final = X_final.skb.apply(skrub.TableVectorizer())

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    predictions = X_final.skb.apply(model, y=y)
    return predictions


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 0.0},
        ),
    )

    # Load the data
    games = pd.read_csv("experiments/scrabble_player_rating/games.csv")
    all_data = pd.read_csv("experiments/scrabble_player_rating/data.csv")

    all_players = all_data.nickname.unique()
    non_bot_players = [player for player in all_players if player not in {"BetterBot", "HastyBot", "STEEBot"}]

    scores = []
    for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
        np.random.seed(seed)
        print(f"Split {split_index}")
        train_players, test_players = train_test_split(non_bot_players, test_size=0.5, random_state=seed)
        train_game_ids = all_data[all_data.nickname.isin(train_players)].game_id.unique()
        test_game_ids = all_data[all_data.nickname.isin(test_players)].game_id.unique()

        train = all_data[all_data.game_id.isin(train_game_ids)]
        test = all_data[all_data.game_id.isin(test_game_ids)]
        test_non_bot_mask = test.nickname.isin(test_players)

        predictions = sempipes_pipeline()
        learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)

        env_train = predictions.skb.get_data()
        env_train["data"] = train
        env_train["games"] = games
        env_train["sempipes_prefitted_state__player_features"] = best_outcome.state

        env_test = predictions.skb.get_data()
        env_test["data"] = test
        env_test["games"] = games
        env_test["sempipes_prefitted_state__player_features"] = best_outcome.state

        learner.fit(env_train)
        y_pred = learner.predict(env_test)

        y_true = test[test_non_bot_mask]["rating"]
        rmse = sqrt(mean_squared_error(y_true, y_pred))  # [test_non_bot_mask]))
        print(f"RMSE on split {split_index}: {rmse}")
        scores.append(rmse)

    print("\nMean final score: ", np.mean(scores), np.std(scores))
