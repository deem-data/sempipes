import pandas as pd
import skrub
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gyyre

games_df = pd.read_csv("comparisons/scrabble-player-rating/games.csv")
turns_df = pd.read_csv("comparisons/scrabble-player-rating/turns.csv")
data_df = pd.read_csv("comparisons/scrabble-player-rating/train.csv")

#data_df = data_df.head(n=500)

games = skrub.var("games", games_df)
turns = skrub.var("turns", turns_df)
data = skrub.var("data", data_df).skb.mark_as_X().skb.set_description("""
Competition featuring data from Woogles.io, where the goal is to predict the ratings of players based on Scrabble gameplay. The evaluation algorithm is RMSE.

The data includes information about thousands of Scrabble games played by three bots on Woogles.io: BetterBot (beginner), STEEBot (intermediate), and HastyBot (advanced). The games are between the bots and their opponents who are regular registered users. You are using metadata about the games as well as turns in each game (i.e., players' racks and where and what they played, AKA gameplay) to predict the rating of players in the test set. 

There is metadata for each game, gameplay data about turns played by each player in each game, and final scores and ratings from BEFORE a given game was played for each player in each game.

Turns data:
 - game_id Unique id for the game
 - turn_number The turn number in the game
 - nickname Player's username on woogles.io
 - rack Player's current rack
 - location Where the player places their turn on the board (NA for games in the test set or if the player didn't make a play, e.g., if they exchanged) move Tiles the player laid (NA for games in the test set; "--" if the turn_type was "Pass"; "(challenge)" if the turn_type was "Challenge"; "-" plus tiles exchanged if the turn_type was "Exchange"; at the end of the game, remaining tiles in a player's rack are in parentheses)
 - points Points the player earned (or lost) in their turn
 - score Player's total score at the time of the turn
 - turn_type Type of turn played ("Play", "Exchange", "Pass", "Six-Zero Rule" (i.e., a game that ends when players pass 3 turns in a row each), "Challenge")

Games data:
 - game_id Unique id for the game
 - first Which player went first
 - time_control_name Name of time control used ("regular", "rapid", or "blitz")
 - game_end_reason How the game ended
 - winner Who won the game
 - created_at When the game was created
 - lexicon English lexicon used in the game ("CSW19", "NWL20", "CSW21")
 - initial_time_seconds Time limit each player has in the game (defines the time control name)
 - increment_seconds Time increment each player gets each time they play a turn
 - rating_mode Whether the game counts towards player ratings or not ("RATED", "CASUAL")
 - max_overtime_minutes How far past the initial time limit players can go before they timeout
 - game_duration_seconds How long the game lasted

Plays data:
 - game_id Unique id for the game
 - nickname Player's username on woogles.io
 - score Final score for each player for each game.
 - rating Player's rating on woogles.io BEFORE the game was played; ratings are per Lexicon / time control name (AKA game variant).
""").skb.subsample(n=100)

def sum_first_five(series):
    return sum(series.values[::-1][:5])


# def replace_winner(row, data):
#     """Set the value of winner to 1 if the player won, -1 if the lost, or 0 if it was a draw."""
#     # Locate opponent as the row with the same game_id but different nickname.
#     opponent_row = data.loc[(data.game_id == row.loc["game_id"]) & (data.nickname != row.loc["nickname"])]
#
#     # Compare scores. Set the winner to 1, the loser to -1 and if a tie, give both 0.
#     if (row.loc["score"] > opponent_row["score"].values).all():
#         row.loc["winner"] = 1
#     elif (row.loc["score"] < opponent_row["score"].values).all():
#         row.loc["winner"] = -1
#     else:
#         row.loc["winner"] = 0
#     return row
#
#
# def replace_first(row):
#     """Set the value in column first to 1 if the player went first in their game,
#     or to 0 if they went second."""
#     if row.loc["first"] == row.loc["nickname"]:
#         row.loc["first"] = 1
#     else:
#         row.loc["first"] = 0
#     return row
#
#
# def relabel_values(data):
#     def relabel(row):
#         row = replace_winner(row, data)
#         row = replace_first(row)
#         return row
#     return relabel


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

y = data.pop("rating").skb.mark_as_y().skb.set_description("the ratings of players based on Scrabble gameplay")
X = data.with_sem_features(
    nl_prompt="""
    Create additional features that could help predict the rating of a player. Such features could relate to how
    the player scores when they go first in a round and to which games they won, how often they won games, etc.
    """,
    name="player_features",
    how_many=10,
)
#new_columns = [column for column in X.columns if column not in data.columns]

#X = data.skb.apply_func(lambda df: df.apply(relabel_values(df), axis=1))



from sklearn.compose import ColumnTransformer

#pass_through_features = ['score', 'total_turns', 'first_five_turns_points', 'max_points_turn', 'min_points_turn', 'max_min_difference', 'first', 'winner', 'initial_time_seconds', 'increment_seconds', 'max_overtime_minutes', 'game_duration_seconds', 'time_used', 'points_per_turn', 'points_per_second']
#nominal_features = ["time_control_name", "game_end_reason", "lexicon", "rating_mode"]
#
#encoder = ColumnTransformer(
#    transformers=[
#        ("ordinal", OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features),
#        ("passthrough", "passthrough", pass_through_features),
#    ]
#)
encoder = skrub.TableVectorizer()
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
    #early_stopping_rounds=1000
)

predictions = X.skb.apply(xgb_reg, y=y)

from sklearn.model_selection import train_test_split
def custom_splitter(all, y, random_state, test_size):
    all_players = all.nickname.unique()
    non_bot_players = [player for player in all_players if player not in {'BetterBot', 'HastyBot', 'STEEBot'}]
    train_players, test_players = train_test_split(non_bot_players, test_size=test_size, random_state=random_state)
    train_game_ids = all[all.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all[all.nickname.isin(test_players)].game_id.unique()

    train = all[all.game_id.isin(train_game_ids)]
    test = all[all.game_id.isin(test_game_ids)]
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
    import numpy as np
    rmse = np.sqrt(mean_squared_error(split["test"]["_skrub_y"], y_pred))
    print(f"RMSE on split {split_index}: {rmse}")
    scores.append(rmse)

print("\nMean final score: ", np.mean(scores), np.std(scores))
