### Competition and Dataset:

 - https://www.kaggle.com/competitions/scrabble-player-rating

### AIDE baseline:

 - https://github.com/WecoAI/aideml/blob/main/sample_results/house-prices-advanced-regression-techniques.py

### Kaggle baseline:

 - https://www.kaggle.com/code/fedi1996/house-prices-data-cleaning-viz-and-modeling#-Modeling--


### Preliminary results

#### AIDE

RMSLE on split 0: 0.1272016552886734
RMSLE on split 1: 0.16392404776501368
RMSLE on split 2: 0.19588776077489128
RMSLE on split 3: 0.12400859880774617
RMSLE on split 4: 0.12094492698517204

Mean final score:  0.1463933979242993 0.02923710836537684


#### Kaggle

RMSLE on 0: 0.12890367223815058
RMSLE on 1: 0.13247316421157498
RMSLE on 2: 0.14341332505317003
RMSLE on 3: 0.12661676220744414
RMSLE on 4: 0.1346140046689381

Mean final score:  0.13320418567585557 0.005807530845031937


#### SemPipes

--- Fitting gyyre.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['AgeAtSale', 'GarageAge', 'IsNew', 'LivAreaRatio', 'NonBedRoomsAbvGr', 'TotalBath', 'TotalBsmtFinSF', 'TotalHouseSF', 'TotalPorchSF', 'YearsSinceRemodel'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtExposure_Ordinal', 'BsmtFinType1_Ordinal', 'BsmtFinType2_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'HeatingQC_Ordinal', 'LandSlope_Ordinal', 'LotShape_Ordinal'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['Age', 'GarageAge', 'IsRemod', 'LivLotRatio', 'TotalBath', 'TotalFinishedSF', 'TotalHouseSF', 'TotalPorchSF', 'TotalRooms', 'YearsSinceRemod'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtExposure_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'FireplaceQu_Ordinal', 'GarageFinish_Ordinal', 'GarageQual_Ordinal', 'HeatingQC_Ordinal', 'KitchenQual_Ordinal'], removed 0 feature columns: []
RMSLE on 0: 0.13024428964198462
--- Fitting gyyre.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 11 new feature columns: ['GarageAge', 'HasBsmt', 'HasFireplace', 'HasGarage', 'HouseAge', 'IsRemodeled', 'RemodAge', 'TotalBathrooms', 'TotalPorchSF', 'TotalRooms', 'TotalSF'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_Ordinal', 'BsmtQual_Ordinal', 'ExterCond_Ordinal', 'ExterQual_Ordinal', 'FireplaceQu_Ordinal', 'GarageFinish_Ordinal', 'HeatingQC_Ordinal', 'KitchenQual_Ordinal', 'LandSlope_Ordinal', 'LotShape_Ordinal'], removed 0 feature columns: []
RMSLE on 1: 0.1324648872296217
--- Fitting gyyre.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['HasFireplace', 'HasGarage', 'HouseAge', 'IsNew', 'IsRemodeled', 'SinceRemodel', 'TotalBath', 'TotalPorchSF', 'TotalRooms', 'TotalSF'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 11 new feature columns: ['BsmtCondOrdinal', 'BsmtQualOrdinal', 'ExterCondOrdinal', 'ExterQualOrdinal', 'FireplaceQuOrdinal', 'GarageCondOrdinal', 'GarageQualOrdinal', 'HeatingQCOrdinal', 'KitchenQualOrdinal', 'LandSlopeOrdinal', 'LotShapeOrdinal'], removed 0 feature columns: []
RMSLE on 2: 0.14429011653187312
--- Fitting gyyre.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['GarageAge', 'HouseAge', 'NewHouse', 'OverallScore', 'Remodeled', 'TotalBath', 'TotalFinishedSF', 'TotalPorchSF', 'TotalRooms', 'YearsSinceRemodel'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCondOrdinal', 'BsmtExposureOrdinal', 'BsmtFinType1Ordinal', 'BsmtFinType2Ordinal', 'BsmtQualOrdinal', 'ExterCondOrdinal', 'ExterQualOrdinal', 'HeatingQCOrdinal', 'LandSlopeOrdinal', 'LotShapeOrdinal'], removed 0 feature columns: []
RMSLE on 3: 0.12817191566033506
--- Fitting gyyre.with_sem_features('Compute additional features from th...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['GarageAge', 'HasFireplace', 'HouseAge', 'IsRemodeled', 'RemodAge', 'TotalBath', 'TotalBsmtFinSF', 'TotalPorchSF', 'TotalRooms', 'TotalSqFeet'], removed 0 feature columns: []
--- Fitting gyyre.with_sem_features('Replace the categorical features (t...', 10)
	> Querying 'openai/gpt-4.1' with 2 messages...'
	> Computed 10 new feature columns: ['BsmtCond_ord', 'BsmtExposure_ord', 'BsmtFinType1_ord', 'BsmtFinType2_ord', 'BsmtQual_ord', 'ExterCond_ord', 'ExterQual_ord', 'GarageFinish_ord', 'GarageQual_ord', 'HeatingQC_ord'], removed 0 feature columns: []
RMSLE on 4: 0.1354389165361233

Mean final score:  0.13412202511998755 0.005626917930257169