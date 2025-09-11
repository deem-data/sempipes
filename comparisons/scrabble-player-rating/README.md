### Competition and Dataset:

 - https://www.kaggle.com/competitions/scrabble-player-rating

### AIDE baseline:

 - https://github.com/WecoAI/aideml/blob/main/sample_results/scrabble-player-rating.py

### Kaggle baseline:

 - https://www.kaggle.com/competitions/scrabble-player-rating




TRAIN SPLIT 44024 1600711342
TEST SPLIT 28398 1032921364
RMSE on split 0: 292.5272125445576
TRAIN SPLIT 59350 2158073668
TEST SPLIT 20735 754240201
RMSE on split 1: 231.66421584540427
TRAIN SPLIT 39952 1448954372
TEST SPLIT 30434 1108799849
RMSE on split 2: 301.1013230243952
TRAIN SPLIT 55562 2021719218
TEST SPLIT 22629 822417426
RMSE on split 3: 244.2488113086773
TRAIN SPLIT 39982 1451525068
TEST SPLIT 30419 1107514501
RMSE on split 4: 286.29052319714947

Mean final score:  271.16641718403673 27.806841479967083


TRAIN SPLIT 44024 1600711342
TEST SPLIT 28398 1032921364
RMSE on split 0: 158.37883529294564
TRAIN SPLIT 59350 2158073668
TEST SPLIT 20735 754240201
RMSE on split 1: 171.50447561579844
TRAIN SPLIT 39952 1448954372
TEST SPLIT 30434 1108799849
RMSE on split 2: 175.95023266537615
TRAIN SPLIT 55562 2021719218
TEST SPLIT 22629 822417426
RMSE on split 3: 192.26843824130367
TRAIN SPLIT 39982 1451525068
TEST SPLIT 30419 1107514501
RMSE on split 4: 161.4045407702336

Mean final score:  171.90130451713148 12.03669931120767


Using python3.12 (3.12.10)
TRAIN SPLIT 44024 1600711342
TEST SPLIT 28398 1032921364
RMSE on split 0: 162.26853707666191
TRAIN SPLIT 59350 2158073668
TEST SPLIT 20735 754240201
RMSE on split 1: 173.90824628234282
TRAIN SPLIT 39952 1448954372
TEST SPLIT 30434 1108799849
RMSE on split 2: 182.7262937770588
TRAIN SPLIT 55562 2021719218
TEST SPLIT 22629 822417426
RMSE on split 3: 194.3937591430085
TRAIN SPLIT 39982 1451525068
TEST SPLIT 30419 1107514501
RMSE on split 4: 167.95912472421676

Mean final score:  176.25119220065775 11.321149222662829




--- Fitting gyyre.with_sem_features('Create additional features that cou...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'zip' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> An error occurred in attempt 2: 'is_first_player'
	> Querying 'openai/gpt-4.1' with 6 messages...'
	> Computed 1 new feature columns: ['is_first_player'], removed 0 feature columns: []
TRAIN SPLIT 44024 1600711342
TEST SPLIT 28398 1032921364
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 9 new feature columns: ['is_first_player', 'player_first_games', 'player_first_prop', 'player_games_won', 'player_mean_score', 'player_timecontrol_games', 'player_total_games', 'player_win_prop', 'score_when_first'], removed 0 feature columns: []
RMSE on split 0: 16.445087030817803
TRAIN SPLIT 59350 2158073668
TEST SPLIT 20735 754240201
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['first_five_turns_frac', 'max_min_diff_per_turn', 'player_mean_score', 'player_points_per_turn_std', 'player_went_first', 'player_win_ratio', 'score_minus_rating', 'score_per_turn', 'won_and_not_first', 'won_and_went_first'], removed 0 feature columns: []
RMSE on split 1: 11.072007743347635
TRAIN SPLIT 39952 1448954372
TEST SPLIT 30434 1108799849
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 14 new feature columns: ['avg_points_per_turn_per_player', 'avg_score_per_player', 'cum_first_count', 'cum_first_rate', 'cum_games', 'cum_win_rate', 'cum_wins', 'first_player_and_win', 'game_avg_score', 'is_first_player', 'opponent_rating', 'player_max_score_ever', 'player_win', 'score_vs_game_avg'], removed 0 feature columns: []
RMSE on split 2: 45.244279194221754
TRAIN SPLIT 55562 2021719218
TEST SPLIT 22629 822417426
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['efficient_scoring', 'first_five_turns_ratio', 'is_first_player', 'moves_per_minute', 'overtime_used', 'points_range_turn', 'rating_mode_encoded', 'score_per_rating', 'time_control_type', 'win'], removed 0 feature columns: []
RMSE on split 3: 20.28021924143003
TRAIN SPLIT 39982 1451525068
TEST SPLIT 30419 1107514501
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'df' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> Computed 1 new feature columns: ['max_min_diff_per_turn'], removed 0 feature columns: []
RMSE on split 4: 20.84680684106681

Mean final score:  22.777680010176805 11.762977152437115

