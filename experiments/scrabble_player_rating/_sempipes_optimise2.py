import warnings

import numpy as np

import sempipes
from experiments.scrabble_player_rating import BOT_NAMES
from experiments.scrabble_player_rating._sempipes_impl2 import sempipes_pipeline2
from sempipes.optimisers import optimise_colopro

warnings.filterwarnings("ignore")

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="openai/gpt-4.1",
        parameters={"temperature": 0.8},
    ),
)


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
            train_players = np.concatenate([player_folds[i] for i in range(self.n_splits) if i != k])
            train_player_game_ids = X[X.nickname.isin(train_players)].game_id.unique()
            train_idx = np.where(X.game_id.isin(train_player_game_ids))[0]

            yield train_idx, valid_idx


pipeline = sempipes_pipeline2("experiments/scrabble_player_rating/validation.csv")

outcomes = optimise_colopro(
    pipeline,
    "player_features",
    num_trials=24,
    scoring="neg_root_mean_squared_error",
    cv=PlayerBasedFolds(5),
    pipeline_definition=sempipes_pipeline2,
)

best_outcome = max(outcomes, key=lambda x: x.score)
print(f"Lowest score: {-1 * best_outcome.score}\n\n")
final_code = "\n".join(best_outcome.state["generated_code"])
print(final_code)
