 - **Performance Evaluation**: Each model's performance was primarily evaluated using RMSLE, with the Gradient Boosting Model performing the best among the tested algorithms. This superiority could be attributed to its effectiveness in learning non-linear relationships and handling outlier impacts well.
- **Model Comparison**: Compared to the baseline models, GBM showed significant improvements with lesser RMSLE values. Neural Networks portrayed potential but required extensive computational resources and fine-tuning.
- **Feature Impact**: Economic factors like inflation rate, unemployment rate, and foreign direct investment were among the top predictors of GDP based on feature importance metrics from models.
- 

➜  sempipes git:(aide_data_integration) ✗ poetry run python -m comparisons.aide_dataintegration.sempipes_impl
--- sempipes.sem_extract_features('['nn']', '{'ic': 'the two letter country code'}', 'Generate the ISO 2 letter country code from the country name')
	> Generated possible columns: [{'feature_name': 'ic', 'feature_prompt': 'the two letter country code', 'input_columns': ['nn']}]
	> Querying 'gemini/gemini-2.5-flash' with 100 requests in batches of size 20...'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:09<00:00,  1.94s/it]
	> Generated 1 columns: ['ic']
--- sempipes.sem_extract_features('['nn']', '{'ic': 'the two letter country code'}', 'Generate the ISO 2 letter country code from the country name')
	> Generated possible columns: [{'feature_name': 'ic', 'feature_prompt': 'the two letter country code', 'input_columns': ['nn']}]
	> Querying 'gemini/gemini-2.5-flash' with 100 requests in batches of size 20...'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:10<00:00,  2.19s/it]
	> Generated 1 columns: ['ic']
RMSE on 0: 0.582163916705152
--- sempipes.sem_extract_features('['nn']', '{'ic': 'the two letter country code'}', 'Generate the ISO 2 letter country code from the country name')
	> Generated possible columns: [{'feature_name': 'ic', 'feature_prompt': 'the two letter country code', 'input_columns': ['nn']}]
	> Querying 'gemini/gemini-2.5-flash' with 100 requests in batches of size 20...'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:10<00:00,  2.06s/it]
	> Generated 1 columns: ['ic']
RMSE on 1: 0.5919013339690259
--- sempipes.sem_extract_features('['nn']', '{'ic': 'the two letter country code'}', 'Generate the ISO 2 letter country code from the country name')
	> Generated possible columns: [{'feature_name': 'ic', 'feature_prompt': 'the two letter country code', 'input_columns': ['nn']}]
	> Querying 'gemini/gemini-2.5-flash' with 100 requests in batches of size 20...'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:11<00:00,  2.21s/it]
	> Generated 1 columns: ['ic']
RMSE on 2: 0.6733200933088405
--- sempipes.sem_extract_features('['nn']', '{'ic': 'the two letter country code'}', 'Generate the ISO 2 letter country code from the country name')
	> Generated possible columns: [{'feature_name': 'ic', 'feature_prompt': 'the two letter country code', 'input_columns': ['nn']}]
	> Querying 'gemini/gemini-2.5-flash' with 100 requests in batches of size 20...'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:08<00:00,  1.72s/it]
	> Generated 1 columns: ['ic']
RMSE on 3: 0.6761268945227444
--- sempipes.sem_extract_features('['nn']', '{'ic': 'the two letter country code'}', 'Generate the ISO 2 letter country code from the country name')
	> Generated possible columns: [{'feature_name': 'ic', 'feature_prompt': 'the two letter country code', 'input_columns': ['nn']}]
	> Querying 'gemini/gemini-2.5-flash' with 100 requests in batches of size 20...'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:10<00:00,  2.07s/it]
	> Generated 1 columns: ['ic']
RMSE on 4: 0.7044388201587518

Mean final score:  0.6455902117329029 0.0491324425164751