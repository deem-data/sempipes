### Competition and Dataset:

 - https://www.kaggle.com/competitions/tmdb-box-office-prediction/

### AIDE baseline:

 - https://github.com/WecoAI/aideml/blob/main/sample_results/tmdb-box-office-prediction.py

### Kaggle baseline:

 - https://www.kaggle.com/code/archfu/notebook95778def47

### Preliminary results

#### AIDE 

```commandline
RMSLE on split 0: 2.1926627824612277
RMSLE on split 1: 2.3359982170198017
RMSLE on split 2: 2.1924068335717295
RMSLE on split 3: 2.208706819042257
RMSLE on split 4: 2.192777714120668

Mean final score:  2.2245104732431367 0.0560912820112512
```

#### Kaggle

```commandline
RMSLE on split 0: 2.1216619040978837
RMSLE on split 1: 2.320811211593718
RMSLE on split 2: 2.1152208389059513
RMSLE on split 3: 2.169924015983668
RMSLE on split 4: 2.1436275872946955

Mean final score:  2.1742491115751834 0.07574507868839137
```

#### Gyyre

```commandline
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'Exception' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> Computed 1 new feature columns: ['num_cast'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'pop_quantiles' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> An error occurred in attempt 2: name 'pop_quantiles' is not defined
	> Querying 'openai/gpt-4.1' with 6 messages...'
	> An error occurred in attempt 3: name '_pop_quantiles' is not defined
	> Querying 'openai/gpt-4.1' with 8 messages...'
	> An error occurred in attempt 4: name 'pop_bucket' is not defined
	> Querying 'openai/gpt-4.1' with 10 messages...'
	> Computed 1 new feature columns: ['popularity_bucket'], removed 0 feature columns: []
RMSLE on split 0: 2.1850046558071323
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'eval' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> Computed 1 new feature columns: ['num_genres'], removed 0 feature columns: []
RMSLE on split 1: 2.336604410940475
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'genre_in_list' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> An error occurred in attempt 2: name 'genre_in_list' is not defined
	> Querying 'openai/gpt-4.1' with 6 messages...'
	> An error occurred in attempt 3: name 'end' is not defined
	> Querying 'openai/gpt-4.1' with 8 messages...'
	> Computed 0 new feature columns: [], removed 0 feature columns: []
RMSLE on split 2: 2.156801770305084
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'Exception' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> Computed 1 new feature columns: ['num_awards_names_in_crew'], removed 0 feature columns: []
RMSLE on split 3: 2.174973483045835
--- Fitting gyyre.with_sem_features('Create additional features that...', 25)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> An error occurred in attempt 1: name 'eval' is not defined
	> Querying 'openai/gpt-4.1' with 4 messages...'
	> Computed 1 new feature columns: ['num_genres'], removed 0 feature columns: []
RMSLE on split 4: 2.1874004215931726

Mean final score:  2.20815694833834 0.06512159300961029
```