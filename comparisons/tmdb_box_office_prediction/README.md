### Competition and Dataset:

 - https://www.kaggle.com/competitions/tmdb-box-office-prediction/

### AIDE baseline:

 - https://github.com/WecoAI/aideml/blob/main/sample_results/tmdb-box-office-prediction.py

### Kaggle baseline:

 - https://www.kaggle.com/code/archfu/notebook95778def47
 -  https://www.kaggle.com/code/dway88/feature-eng-feature-importance-random-forest
 - 
### Preliminary results

#### Copilot

RMSLE on split 0: 2.319964225830093
RMSLE on split 1: 2.3173161589719506
RMSLE on split 2: 2.498929222736798
RMSLE on split 3: 2.419895734397335
RMSLE on split 4: 2.587022864398839

Mean final score:  2.428625641267003 0.10421668694681777

#### AIDE 

```commandline
RMSLE on split 0: 2.2662804402480528
RMSLE on split 1: 2.2762388818076813
RMSLE on split 2: 2.4865292057138797
RMSLE on split 3: 2.227291588321042
RMSLE on split 4: 2.5228661124137584

Mean final score:  2.3558412457008826 0.12317413496025241
```

#### Kaggle

```commandline
RMSLE on split 0: 2.0356605930236236
RMSLE on split 1: 2.0664315538831057
RMSLE on split 2: 2.3993225524670536
RMSLE on split 3: 2.3293931819757145
RMSLE on split 4: 2.3817040681718584

Mean final score:  2.242502389904271 0.15830612660256735
```

#### SemPipes

```commandline
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'Exception' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> Computed 1 new feature columns: ['num_directors'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'eval' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> Computed 1 new feature columns: ['num_genres'], removed 0 feature columns: []
RMSLE on split 0: 2.1102837524146887
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 36 new feature columns: ['has_collection', 'has_famous_director', 'has_homepage', 'has_tagline', 'is_action', 'is_adventure', 'is_animation', 'is_comedy', 'is_crime', 'is_documentary', 'is_drama', 'is_english', 'is_family', 'is_fantasy', 'is_foreign', 'is_history', 'is_horror', 'is_music', 'is_mystery', 'is_romance', 'is_scifi', 'is_thriller', 'is_war', 'is_western', 'log_budget', 'main_genre', 'num_cast', 'num_crew', 'num_genres', 'num_keywords', 'num_production_companies', 'num_production_countries', 'num_spoken_languages', 'release_month', 'release_year', 'runtime_missing'], removed 0 feature columns: []
RMSLE on split 1: 2.094021567619216
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 26 new feature columns: ['budget_log1p', 'director_name', 'director_popularity', 'has_collection', 'has_homepage', 'has_imdb_id', 'has_known_star', 'has_tagline', 'is_english', 'is_released', 'is_sequel', 'is_summer_release', 'main_genre', 'num_cast', 'num_crew', 'num_genres', 'num_keywords', 'num_production_companies', 'num_production_countries', 'num_spoken_languages', 'overview_length', 'popularity_log1p', 'release_dayofweek', 'release_month', 'release_year', 'runtime_log1p'], removed 0 feature columns: []
RMSLE on split 2: 2.417853387666705
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 25 new feature columns: ['cast_popularity', 'director_count', 'has_collection', 'has_homepage', 'has_keyword_franchise', 'has_poster', 'has_tagline', 'is_english', 'is_released', 'is_summer_release', 'log_budget', 'num_cast', 'num_crew', 'num_genres', 'num_keywords', 'num_production_companies', 'num_production_countries', 'num_spoken_languages', 'overview_length', 'release_dayofweek', 'release_month', 'release_year', 'runtime_missing', 'top_production_company_major', 'top_production_country_us'], removed 0 feature columns: []
RMSLE on split 3: 2.3189431569845635
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 25 new feature columns: ['cast_top_gender_female', 'has_budget', 'has_collection', 'has_homepage', 'has_poster', 'has_tagline', 'is_english', 'is_multicompany', 'is_multicountry', 'is_multigenre', 'is_multilang', 'is_released', 'log_budget', 'num_cast', 'num_crew', 'num_genres', 'num_keywords', 'num_production_companies', 'num_production_countries', 'num_spoken_languages', 'overview_length', 'release_dayofweek', 'release_month', 'release_year', 'runtime_missing'], removed 0 feature columns: []
RMSLE on split 4: 2.41025955023866

Mean final score:  2.2702722829847666 0.14170823265364602
```

#### SemPipes (optimised)

```commandline
--- Fitting gyyre.with_sem_features('Create additional features...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 29 new feature columns: ['budget_log', 'has_collection', 'has_female_director', 'has_female_writer', 'has_homepage', 'has_known_franchise', 'has_oscar_winner_cast', 'has_tagline', 'is_action', 'is_animation', 'is_comedy', 'is_drama', 'is_english', 'is_horror', 'is_romance', 'is_sequel', 'num_cast', 'num_crew', 'num_genres', 'num_keywords', 'num_production_companies', 'num_production_countries', 'num_spoken_languages', 'num_top_billed_cast', 'popularity_log', 'release_dayofweek', 'release_month', 'release_year', 'runtime_log'], removed 0 feature columns: []
--- Using provided state for gyyre.with_sem_features('Create additional features...', 25)
RMSLE on split 0: 2.030810021840814
--- Using provided state for gyyre.with_sem_features('Create additional features...', 25)
RMSLE on split 1: 2.0139658178310804
--- Using provided state for gyyre.with_sem_features('Create additional features...', 25)
RMSLE on split 2: 2.3715216643128656
--- Using provided state for gyyre.with_sem_features('Create additional features...', 25)
RMSLE on split 3: 2.2577813610215705
--- Using provided state for gyyre.with_sem_features('Create additional features...', 25)
RMSLE on split 4: 2.366580317844579

Mean final score:  2.208131836570182 0.15710550666230977
```

Kaggle 2
```commandline
RMSLE on split 0: 2.080288206673187
RMSLE on split 1: 2.0516200198513816
RMSLE on split 2: 2.368225270402235
RMSLE on split 3: 2.2406475699121695
RMSLE on split 4: 2.367719034182676

Mean final score:  2.22170002020433 0.13570167173246156
```
Kaggle 2 - optimised
```commandline
RMSLE on split 0: 2.068715096489106
RMSLE on split 1: 2.0185962194681943
RMSLE on split 2: 2.3402040963942525
RMSLE on split 3: 2.2650815075545676
RMSLE on split 4: 2.3675541971533294

Mean final score:  2.2120302234118903 0.14239847355881113
```