### Competition and Dataset:

 - https://www.kaggle.com/competitions/scrabble-player-rating

### AIDE baseline:

 - https://github.com/WecoAI/aideml/blob/main/sample_results/scrabble-player-rating.py

### Kaggle baseline:

 - https://www.kaggle.com/code/mikepenkov/rating-scrabble-players-with-xgb-regression


### Preliminary results

#### Copilot with context

It failed twice before producing working code: (1) KeyError: "Column(s) ['move_score'] do not exist"; (2) TypeError: OneHotEncoder.init() got an unexpected keyword argument 'sparse'

RMSE on split 0: 240.1909288384617
RMSE on split 1: 180.45831506877656
RMSE on split 2: 203.16180623362172
RMSE on split 3: 177.06742644034625
RMSE on split 4: 168.2350395474886

Mean final score:  193.82270322573896 25.889402831817236

#### AIDE

```commandline
RMSE on split 0: 283.30835226797274
RMSE on split 1: 223.05487796629538
RMSE on split 2: 283.8207560311914
RMSE on split 3: 250.62215926158237
RMSE on split 4: 234.15469769749097

Mean final score:  254.9921686449066 24.924612009211693
```

#### Kaggle

```commandline
RMSE on split 0: 213.07205170728045
RMSE on split 1: 202.8428906352155
RMSE on split 2: 155.5321510170807
RMSE on split 3: 171.10543624473186
RMSE on split 4: 158.08040379028643

Mean final score:  180.126586678919 23.553070848657562
```


#### SemPipes

```commandline
--- Fitting sempipes.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['avg_points_per_turn', 'first_five_points_ratio', 'is_first_player', 'max_min_diff_per_turn', 'player_avg_first_five_points', 'player_avg_game_duration', 'player_avg_max_min_diff', 'player_avg_max_points_turn', 'player_avg_min_points_turn', 'player_avg_points_per_second', 'player_avg_points_per_turn', 'player_avg_score', 'player_avg_turns', 'player_win_rate', 'points_per_second'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 16 new feature columns: ['first_five_points_frac', 'game_avg_score', 'is_first', 'is_winner', 'max_min_diff_per_turn', 'player_avg_points_per_second', 'player_avg_points_per_turn', 'player_avg_score', 'player_win_rate', 'points_per_second', 'points_per_turn', 'score_vs_game_avg', 'seconds_per_turn', 'turns_per_minute', 'win_when_first', 'win_when_not_first'], removed 0 feature columns: []
RMSE on split 0: 188.2115041980697
--- Fitting sempipes.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['is_first_player', 'player_avg_first5', 'player_avg_game_duration', 'player_avg_increment', 'player_avg_initial_time', 'player_avg_max_min_diff', 'player_avg_max_points_turn', 'player_avg_min_points_turn', 'player_avg_score', 'player_avg_turns', 'player_first_win_rate', 'player_notfirst_win_rate', 'player_win_rate', 'points_per_second', 'points_per_turn'], removed 0 feature columns: []
RMSE on split 1: 127.99819182072456
--- Fitting sempipes.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['avg_points_per_turn', 'avg_time_per_turn', 'first_five_turns_ratio', 'game_end_reason_code', 'is_first_player', 'is_rated', 'lexicon_code', 'max_min_diff_per_turn', 'player_avg_score', 'player_win', 'points_per_second', 'time_control_code', 'turns_per_minute', 'win_as_first', 'win_as_second'], removed 0 feature columns: []
RMSE on split 2: 176.18526625934135
--- Fitting sempipes.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 136 new feature columns: ['avg_points_per_turn', 'avg_turn_duration', 'duration_if_first', 'duration_if_notfirst', 'first5_if_first', 'first5_if_notfirst', 'first_five_points_frac', 'increment_if_first', 'increment_if_notfirst', 'initial_time_if_first', 'initial_time_if_notfirst', 'is_first_player', 'is_winner', 'max_min_diff_frac', 'max_turn_if_first', 'max_turn_if_notfirst', 'maxmin_diff_if_first', 'maxmin_diff_if_notfirst', 'min_turn_if_first', 'min_turn_if_notfirst', 'overtime_if_first', 'overtime_if_notfirst', 'player_avg_duration_first_diff', 'player_avg_duration_if_first', 'player_avg_duration_if_notfirst', 'player_avg_first5', 'player_avg_first5_first_diff', 'player_avg_first5_if_first', 'player_avg_first5_if_notfirst', 'player_avg_game_duration', 'player_avg_increment', 'player_avg_increment_first_diff', 'player_avg_increment_if_first', 'player_avg_increment_if_notfirst', 'player_avg_initial_time', 'player_avg_initial_time_first_diff', 'player_avg_initial_time_if_first', 'player_avg_initial_time_if_notfirst', 'player_avg_max_turn', 'player_avg_max_turn_first_diff', 'player_avg_max_turn_if_first', 'player_avg_max_turn_if_notfirst', 'player_avg_maxmin_diff', 'player_avg_maxmin_diff_first_diff', 'player_avg_maxmin_diff_if_first', 'player_avg_maxmin_diff_if_notfirst', 'player_avg_min_turn', 'player_avg_min_turn_first_diff', 'player_avg_min_turn_if_first', 'player_avg_min_turn_if_notfirst', 'player_avg_overtime', 'player_avg_overtime_first_diff', 'player_avg_overtime_if_first', 'player_avg_overtime_if_notfirst', 'player_avg_points_per_second', 'player_avg_points_per_second_first_diff', 'player_avg_points_per_second_if_first', 'player_avg_points_per_second_if_notfirst', 'player_avg_points_per_turn_cumsum', 'player_avg_points_per_turn_first_diff', 'player_avg_points_per_turn_hist', 'player_avg_points_per_turn_if_first', 'player_avg_points_per_turn_if_notfirst', 'player_avg_score', 'player_avg_score_first_diff', 'player_avg_score_if_first', 'player_avg_score_if_notfirst', 'player_avg_total_turns', 'player_avg_total_turns_first_diff', 'player_avg_total_turns_if_first', 'player_avg_total_turns_if_notfirst', 'player_avg_turn_duration', 'player_avg_turn_duration_first_diff', 'player_avg_turn_duration_if_first', 'player_avg_turn_duration_if_notfirst', 'player_duration_if_first_cumsum', 'player_duration_if_notfirst_cumsum', 'player_first5_cumsum', 'player_first5_if_first_cumsum', 'player_first5_if_notfirst_cumsum', 'player_first_games', 'player_first_win_rate', 'player_first_wins', 'player_frac_first', 'player_frac_notfirst', 'player_game_count', 'player_game_duration_cumsum', 'player_increment_cumsum', 'player_increment_if_first_cumsum', 'player_increment_if_notfirst_cumsum', 'player_initial_time_cumsum', 'player_initial_time_if_first_cumsum', 'player_initial_time_if_notfirst_cumsum', 'player_max_turn_cumsum', 'player_max_turn_if_first_cumsum', 'player_max_turn_if_notfirst_cumsum', 'player_maxmin_diff_cumsum', 'player_maxmin_diff_if_first_cumsum', 'player_maxmin_diff_if_notfirst_cumsum', 'player_min_turn_cumsum', 'player_min_turn_if_first_cumsum', 'player_min_turn_if_notfirst_cumsum', 'player_notfirst_games', 'player_notfirst_win_rate', 'player_notfirst_wins', 'player_overtime_cumsum', 'player_overtime_if_first_cumsum', 'player_overtime_if_notfirst_cumsum', 'player_points_per_second_cumsum', 'player_points_per_second_if_first_cumsum', 'player_points_per_second_if_notfirst_cumsum', 'player_points_per_turn_if_first_cumsum', 'player_points_per_turn_if_notfirst_cumsum', 'player_score_cumsum', 'player_score_if_first_cumsum', 'player_score_if_notfirst_cumsum', 'player_total_turns_cumsum', 'player_total_turns_if_first_cumsum', 'player_total_turns_if_notfirst_cumsum', 'player_turn_duration_cumsum', 'player_turn_duration_if_first_cumsum', 'player_turn_duration_if_notfirst_cumsum', 'player_win_cumsum', 'player_win_rate', 'player_win_rate_first_diff', 'points_per_second', 'points_per_second_if_first', 'points_per_second_if_notfirst', 'points_per_turn_if_first', 'points_per_turn_if_notfirst', 'score_if_first', 'score_if_notfirst', 'total_turns_if_first', 'total_turns_if_notfirst', 'turn_duration_if_first', 'turn_duration_if_notfirst'], removed 0 feature columns: []
RMSE on split 3: 238.03206848027853
--- Fitting sempipes.with_sem_features('Create additional features that cou...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['player_avg_first5', 'player_avg_game_duration', 'player_avg_max_min_diff', 'player_avg_max_points_turn', 'player_avg_min_points_turn', 'player_avg_points_per_sec', 'player_avg_points_per_turn', 'player_avg_points_per_turn_first', 'player_avg_score', 'player_avg_score_first', 'player_avg_total_turns', 'player_games_first', 'player_games_played', 'player_win_rate', 'player_win_rate_first'], removed 0 feature columns: []
RMSE on split 4: 117.569718345648

Mean final score:  169.59934982081242 43.613214665771835
```

#### SemPipes (opt)
```commandline
--- Fitting sempipes.with_sem_features('Create additional features that...', 15)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 15 new feature columns: ['avg_points_first_five_turns', 'finished_early', 'first_five_points_ratio', 'initial_time_per_turn', 'max_to_avg_turn_points', 'maxmin_diff_to_avg_turn_points', 'min_to_avg_turn_points', 'points_per_second', 'points_per_turn', 'score_per_overtime_minute', 'seconds_per_turn', 'turns_per_minute', 'used_overtime', 'went_first', 'won_game'], removed 0 feature columns: []
--- Using provided state for sempipes.with_sem_features('Create additional features that...', 15)
RMSE on split 0: 202.20882808683206
--- Using provided state for sempipes.with_sem_features('Create additional features that...', 15)
RMSE on split 1: 124.04541758968769
--- Using provided state for sempipes.with_sem_features('Create additional features that...', 15)
RMSE on split 2: 176.2383750598887
--- Using provided state for sempipes.with_sem_features('Create additional features that...', 15)
RMSE on split 3: 128.09396727842025
--- Using provided state for sempipes.with_sem_features('Create additional features that...', 15)
RMSE on split 4: 128.4171744866122

Mean final score:  151.80075250028818 31.677542464595522
```

mini-swe-agent
```console
RMSE on split 0: 233.7602505283591
RMSE on split 1: 207.74902351945312
RMSE on split 2: 245.44380146509627
RMSE on split 3: 221.18096561510455
RMSE on split 4: 234.4994609928796

Mean final score:  228.52670042417853 12.92225909487207
```