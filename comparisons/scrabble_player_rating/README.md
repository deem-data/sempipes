### Competition and Dataset:

 - https://www.kaggle.com/competitions/scrabble-player-rating

### AIDE baseline:

 - https://github.com/WecoAI/aideml/blob/main/sample_results/scrabble-player-rating.py

### Kaggle baseline:

 - https://www.kaggle.com/competitions/scrabble-player-rating


### Preliminary results

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

#### AIDE

```commandline
RMSE on split 0: 292.5272125445576
RMSE on split 1: 231.66421584540427
RMSE on split 2: 301.1013230243952
RMSE on split 3: 244.2488113086773
RMSE on split 4: 286.29052319714947

Mean final score:  271.16641718403673 27.806841479967083
```

#### Kaggle

```commandline
RMSE on split 0: 158.37883529294564
RMSE on split 1: 171.50447561579844
RMSE on split 2: 175.95023266537615
RMSE on split 3: 192.26843824130367
RMSE on split 4: 161.4045407702336

Mean final score:  171.90130451713148 12.03669931120767
```


#### Gyyre

```commandline
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['first_five_turns_ratio', 'game_is_rated', 'is_regular_time_control', 'loser_points', 'max_min_turn_ratio', 'overtime_per_turn', 'overtime_used_ratio', 'player_is_first', 'points_per_second', 'points_per_turn', 'resigned_game', 'score_diff_max_min', 'score_per_initial_time', 'turn_duration_avg', 'winner_points'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 16 new feature columns: ['avg_points_per_turn', 'first_turn_points_ratio', 'high_initial_lead', 'is_first', 'is_rapid_time', 'max_min_turn_diff_ratio', 'max_points_turn_ratio', 'min_points_turn_ratio', 'overtime_used', 'player_won', 'points_late_game', 'points_late_per_turn', 'points_per_second', 'relative_first', 'turns_late_game', 'turns_per_minute'], removed 0 feature columns: []
RMSE on split 0: 168.87156500947694
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 16 new feature columns: ['first_five_ratio', 'is_first', 'is_lexicon_csw', 'is_rated', 'is_winner', 'max_points_perc', 'maxmin_diff_perc', 'min_points_perc', 'score_per_second', 'score_per_turn', 'seconds_per_turn', 'time_use_perc', 'total_max_time_seconds', 'turns_per_min', 'win_as_first', 'win_as_second'], removed 0 feature columns: []
RMSE on split 1: 171.64270097174247
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['first_and_winner', 'first_five_turns_frac', 'game_duration_minutes', 'is_rapid', 'is_resigned', 'is_winner', 'play_time_fraction', 'points_per_second', 'points_per_turn', 'score_by_max_turn', 'score_by_min_turn', 'turn_point_range', 'turns_per_minute', 'went_first', 'win_rated'], removed 0 feature columns: []
RMSE on split 2: 165.003693140487
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['avg_points_per_turn_nickname', 'avg_score_nickname', 'avg_seconds_per_turn_nickname', 'first5_frac_total', 'first5_points_per_turn_normalized', 'fraction_first_nickname', 'games_played', 'is_first', 'is_winner', 'point_range_game', 'points_per_second', 'points_per_turn', 'score_when_first', 'seconds_per_turn', 'win_rate_nickname'], removed 0 feature columns: []
RMSE on split 3: 151.534230611024
--- Fitting gyyre.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['avg_points_per_turn', 'did_win', 'first_five_points_ratio', 'game_end_resigned', 'game_end_standard', 'is_first_player', 'is_rapid_time_control', 'is_rated', 'is_regular_time_control', 'max_min_points_gap', 'player_avg_points_per_second', 'player_avg_points_per_turn', 'player_avg_score', 'player_win_rate', 'points_per_second'], removed 0 feature columns: []
RMSE on split 4: 128.30979052180703

Mean final score:  157.07239605090749 15.952293881347032
```