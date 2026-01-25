import sempipes
import skrub    
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sempipes.optimisers import optimise_colopro
from sempipes.optimisers.evolutionary_search import EvolutionarySearch
from experiments.scrabble_player_rating import BOT_NAMES
import numpy as np

def sempipes_pipeline():
    games = skrub.var("games")
    turns = skrub.var("turns")
    data = skrub.var("data").skb.mark_as_X()
    # Merge the datasets on game_id
    merged_data = data.join(games, on="game_id", rsuffix="_games")
    merged_data = merged_data.join(
        turns.groupby("game_id").agg({"points": "sum"}).reset_index(),
        on="game_id", rsuffix="_turns"
    )

    y = merged_data["rating"].skb.mark_as_y()
    X = merged_data.drop(columns=["rating"])

    X = X.sem_gen_features(
        nl_prompt="""
        Create additional features that could help predict the rating of a player. Such features could relate to how
        the player scores when they go first in a round and to which games they won, how often they won games, etc.
    
        Furthermore, consider how much time a player typically uses, how many points per turn they make, how many points per second, etc.
    
        """,
        name="player_features",
        how_many=15,
    )

    X = X.skb.apply_func(lambda x: x.replace([float("inf"), -float("inf")], 0))
    X = X.skb.apply(skrub.TableVectorizer())

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    predictions = X.skb.apply(model, y=y)

    return predictions


class PlayerBasedFolds:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):  # pylint: disable=unused-argument
        return self.n_splits

    def __str__(self):
        return f"player-based-{self.n_splits}"

    def split(self, X, y=None, groups=None):  # pylint: disable=unused-argument
        all_players = X.nickname.unique()
        non_bot_players = [player for player in all_players if player not in BOT_NAMES]

        # Partition non-bot players into n_splits folds
        player_folds = np.array_split(non_bot_players, self.n_splits)

        for k in range(self.n_splits):
            valid_players = player_folds[k]
            valid_player_game_ids = X[X.nickname.isin(valid_players)].game_id.unique()
            valid_idx = np.where(X.game_id.isin(valid_player_game_ids))[0]
            # valid_idx = np.where(
            #     (X.game_id.isin(valid_player_game_ids)) & (X.nickname.isin(valid_players))
            # )[0]
            train_players = np.concatenate([player_folds[i] for i in range(self.n_splits) if i != k])
            train_player_game_ids = X[X.nickname.isin(train_players)].game_id.unique()
            train_idx = np.where(X.game_id.isin(train_player_game_ids))[0]

            yield train_idx, valid_idx    

if __name__ == "__main__":

    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
    )

    # Load the data
    games = pd.read_csv("experiments/scrabble_player_rating/games.csv")
    turns = pd.read_csv("experiments/scrabble_player_rating/turns.csv.gz")
    all_data = pd.read_csv("experiments/scrabble_player_rating/validation.csv")

    pipeline = sempipes_pipeline()

    outcomes = optimise_colopro(
        pipeline,
        "player_features",
        num_trials=24,
        search=EvolutionarySearch(population_size=6),
        scoring="neg_root_mean_squared_error",
        cv=PlayerBasedFolds(5),
        additional_env_variables={"games": games, "turns": turns, "data": all_data},
    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print("\n".join(best_outcome.state["generated_code"]))
