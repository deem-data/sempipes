from sklearn.model_selection import train_test_split

BOT_NAMES = {"BetterBot", "HastyBot", "MasterBot", "STEEBot"}


def custom_splitter(all_data, y, random_state, test_size):  # pylint: disable=unused-argument
    all_players = all_data.nickname.unique()
    non_bot_players = [player for player in all_players if player not in BOT_NAMES]
    train_players, test_players = train_test_split(non_bot_players, test_size=test_size, random_state=random_state)
    train_game_ids = all_data[all_data.nickname.isin(train_players)].game_id.unique()
    test_game_ids = all_data[all_data.nickname.isin(test_players)].game_id.unique()

    train = all_data[all_data.game_id.isin(train_game_ids)]
    test = all_data[all_data.game_id.isin(test_game_ids)]

    return train, test, train.rating, test.rating


__all__ = ["BOT_NAMES", "custom_splitter"]
