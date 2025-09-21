### Competition and Dataset:

 - https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

### AIDE baseline:

 - https://github.com/WecoAI/aideml/blob/main/sample_results/house-prices-advanced-regression-techniques.py

### Kaggle baseline:

 - https://www.kaggle.com/code/fedi1996/house-prices-data-cleaning-viz-and-modeling#-Modeling--


### Preliminary results

#### AIDE

RMSLE on split 0: 0.1882568531459395
RMSLE on split 1: 0.16358993661193233
RMSLE on split 2: 0.16676931781796867
RMSLE on split 3: 0.24140131100249435
RMSLE on split 4: 0.16953173782447592

Mean final score:  0.18590983128056215 0.02904338731020512


#### Kaggle

RMSLE on 0: 0.15036263705270533
RMSLE on 1: 0.1634540031318504
RMSLE on 2: 0.18289421458593283
RMSLE on 3: 0.1802702404213525
RMSLE on 4: 0.147736554689603

Mean final score:  0.16494352997628883 0.014615308811021823


#### SemPipes

Using python3.12 (3.12.10)
--- Fitting sempipes.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['HasFireplace', 'HasGarage', 'HouseAge', 'IsRemodeled', 'TotalBath', 'TotalFinishedSF', 'TotalHomeQual', 'TotalPorchSF', 'TotalRooms', 'YearsSinceRemodel'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtExposure_Ordinal', 'BsmtFinType1_Ordinal', 'BsmtFinType2_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'HeatingQC_Ordinal', 'LandSlope_Ordinal', 'LotShape_Ordinal'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['HasFireplace', 'HasGarage', 'HouseAge', 'IsNew', 'IsRemodeled', 'RemodAge', 'TotalBathrooms', 'TotalHouseSF', 'TotalPorchSF', 'TotalRooms'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_ordinal', 'BsmtQual_ordinal', 'ExterCond_ordinal', 'ExterQual_ordinal', 'FireplaceQu_ordinal', 'GarageQual_ordinal', 'HeatingQC_ordinal', 'KitchenQual_ordinal', 'LandSlope_ordinal', 'LotShape_ordinal'], removed 0 feature columns: []
RMSLE on 0: 0.15153676089952559
--- Fitting sempipes.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['GarageAge', 'HasFireplace', 'HouseAge', 'IsNew', 'IsRemodeled', 'RemodAge', 'TotalBathrooms', 'TotalHouseSF', 'TotalPorchSF', 'TotalRooms'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_ordinal', 'BsmtExposure_ordinal', 'BsmtFinType1_ordinal', 'BsmtFinType2_ordinal', 'BsmtQual_ordinal', 'ExterCond_ordinal', 'ExterQual_ordinal', 'HeatingQC_ordinal', 'LandSlope_ordinal', 'LotShape_ordinal'], removed 0 feature columns: []
RMSLE on 1: 0.16014809675037128
--- Fitting sempipes.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['GarageAge', 'HasFireplace', 'HasGarage', 'HouseAge', 'IsRemodeled', 'RemodAge', 'TotalBathrooms', 'TotalHouseSF', 'TotalPorchSF', 'TotalRooms'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtExposure_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'FireplaceQu_Ordinal', 'HeatingQC_Ordinal', 'KitchenQual_Ordinal', 'LandSlope_Ordinal', 'LotShape_Ordinal'], removed 0 feature columns: []
RMSLE on 2: 0.18595017386929263
--- Fitting sempipes.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['GarageAge', 'HasFireplace', 'HasGarage', 'HouseAge', 'IsRemodeled', 'RemodAge', 'TotalBathrooms', 'TotalHouseSF', 'TotalPorchSF', 'TotalRooms'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'FireplaceQu_Ordinal', 'GarageQual_Ordinal', 'HeatingQC_Ordinal', 'KitchenQual_Ordinal', 'LandSlope_Ordinal', 'LotShape_Ordinal'], removed 0 feature columns: []
RMSLE on 3: 0.1815138224420401
--- Fitting sempipes.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['GarageAge', 'HasFireplace', 'HasGarage', 'HouseAge', 'IsRemodeled', 'RemodAge', 'TotalBathrooms', 'TotalHouseSF', 'TotalPorchSF', 'TotalRooms'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_ordinal', 'BsmtExposure_ordinal', 'BsmtFinType1_ordinal', 'BsmtFinType2_ordinal', 'BsmtQual_ordinal', 'ExterCond_ordinal', 'ExterQual_ordinal', 'HeatingQC_ordinal', 'LandSlope_ordinal', 'LotShape_ordinal'], removed 0 feature columns: []
RMSLE on 4: 0.14615830918345296

Mean final score:  0.16506143262893652 0.015946199120843704

#### SemPipes (opt)

--- Fitting sempipes.with_sem_features('Compute additional features fro...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['GarageAge', 'HasFireplace', 'HasGarage', 'HouseAge', 'IsRemodeled', 'RemodAge', 'TotalBathrooms', 'TotalHouseSF', 'TotalPorchSF', 'TotalRooms'], removed 0 feature columns: []
--- Fitting sempipes.with_sem_features('Replace the categorical feature...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtExposure_Ordinal', 'BsmtFinType1_Ordinal', 'BsmtFinType2_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'HeatingQC_Ordinal', 'LandSlope_Ordinal', 'LotShape_Ordinal'], removed 0 feature columns: []
--- Using provided state for sempipes.with_sem_features('Compute additional features fro...', 10)
--- Fitting sempipes.with_sem_features('Replace the categorical feature...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_ordinal', 'BsmtExposure_ordinal', 'BsmtQual_ordinal', 'ExterCond_ordinal', 'ExterQual_ordinal', 'FireplaceQu_ordinal', 'HeatingQC_ordinal', 'KitchenQual_ordinal', 'LandSlope_ordinal', 'LotShape_ordinal'], removed 0 feature columns: []
RMSLE on split 0: 0.15045404854031866
--- Using provided state for sempipes.with_sem_features('Compute additional features fro...', 10)
--- Fitting sempipes.with_sem_features('Replace the categorical feature...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ord', 'BsmtQual_Ord', 'ExterCond_Ord', 'ExterQual_Ord', 'FireplaceQu_Ord', 'GarageCond_Ord', 'GarageQual_Ord', 'HeatingQC_Ord', 'KitchenQual_Ord', 'PavedDrive_Ord'], removed 0 feature columns: []
RMSLE on split 1: 0.15360205455262013
--- Using provided state for sempipes.with_sem_features('Compute additional features fro...', 10)
--- Fitting sempipes.with_sem_features('Replace the categorical feature...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_ordinal', 'BsmtExposure_ordinal', 'BsmtFinType1_ordinal', 'BsmtFinType2_ordinal', 'BsmtQual_ordinal', 'ExterCond_ordinal', 'ExterQual_ordinal', 'HeatingQC_ordinal', 'LandSlope_ordinal', 'LotShape_ordinal'], removed 0 feature columns: []
RMSLE on split 2: 0.1773381506668857
--- Using provided state for sempipes.with_sem_features('Compute additional features fro...', 10)
--- Fitting sempipes.with_sem_features('Replace the categorical feature...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'FireplaceQu_Ordinal', 'GarageQual_Ordinal', 'HeatingQC_Ordinal', 'KitchenQual_Ordinal', 'LandSlope_Ordinal', 'LotShape_Ordinal'], removed 0 feature columns: []
RMSLE on split 3: 0.17316456406226807
--- Using provided state for sempipes.with_sem_features('Compute additional features fro...', 10)
--- Fitting sempipes.with_sem_features('Replace the categorical feature...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_ordinal', 'BsmtExposure_ordinal', 'BsmtFinType1_ordinal', 'BsmtFinType2_ordinal', 'BsmtQual_ordinal', 'ExterCond_ordinal', 'ExterQual_ordinal', 'HeatingQC_ordinal', 'LandSlope_ordinal', 'LotShape_ordinal'], removed 0 feature columns: []
RMSLE on split 4: 0.13941528978534323

Mean final score:  0.15879482152148716 0.01429995894757375