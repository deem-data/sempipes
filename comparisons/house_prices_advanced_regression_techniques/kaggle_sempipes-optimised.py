import json
import warnings

import numpy as np

import sempipes  # pylint: disable=unused-import
from comparisons.house_prices_advanced_regression_techniques._sempipes_impl import rmsle, sempipes_pipeline

warnings.filterwarnings("ignore")

pipeline = sempipes_pipeline("comparisons/house_prices_advanced_regression_techniques/data.csv")

with open("comparisons/house_prices_advanced_regression_techniques/_sempipes_state.json", "r", encoding="utf-8") as f:
    state = json.load(f)

state = {
    "generated_code": [
        """
# (Total number of bathrooms)
# Usefulness: The total number of bathrooms is a significant factor in determining house value. Combining full and half baths from both the basement and above grade provides a comprehensive measure of a home's convenience and capacity, which directly correlates with its price.
# Input samples: {'BsmtFullBath': [1, 1, 0], 'BsmtHalfBath': [0, 0, 0], 'FullBath': [1, 2, 1], 'HalfBath': [0, 1, 0]}
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

# (Total square footage of the house)
# Usefulness: The total living area is one of the most critical predictors of a house's price. This feature combines the above-grade living area with the total basement area to give a complete picture of the house's size.
# Input samples: {'GrLivArea': [1144, 2520, 1520], 'TotalBsmtSF': [864, 1338, 793]}
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']

# (Age of the house at the time of sale)
# Usefulness: The age of a house is a key determinant of its value, reflecting factors like depreciation, architectural style, and potential need for repairs. Newer houses generally command higher prices.
# Input samples: {'YrSold': [2009, 2006, 2009], 'YearBuilt': [1961, 1993, 1932]}
df['HouseAge'] = df['YrSold'] - df['YearBuilt']

# (Age of the remodel at the time of sale)
# Usefulness: This feature captures how recently a house was remodeled. A recent remodel can significantly increase a property's value, making this a stronger predictor than just the original construction year. A value of 0 indicates a remodel in the year of sale.
# Input samples: {'YrSold': [2009, 2006, 2009], 'YearRemodAdd': [1983, 1993, 2000]}
df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']

# (Total square footage of all porch areas)
# Usefulness: Porches add to the usable outdoor living space of a home. A larger total porch area can increase a home's appeal and, consequently, its market value.
# Input samples: {'OpenPorchSF': [0, 55, 0], 'EnclosedPorch': [0, 0, 56], '3SsnPorch': [0, 0, 0], 'ScreenPorch': [0, 0, 0]}
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

# (Combined overall quality and condition score)
# Usefulness: This interaction feature amplifies the effect of both overall quality and condition. A house with high quality and in good condition will have a much higher score, likely correlating strongly with a higher sale price, compared to a house that is good in only one of these aspects.
# Input samples: {'OverallQual': [5, 8, 5], 'OverallCond': [6, 7, 5]}
df['OverallQualCond'] = df['OverallQual'] * df['OverallCond']

# (Binary feature indicating if the house was ever remodeled)
# Usefulness: A simple flag for whether a remodel has occurred can be a powerful signal. It indicates that an investment has been made to update the property, which typically increases its value.
# Input samples: {'YearBuilt': [1961, 1993, 1932], 'YearRemodAdd': [1983, 1993, 2000]}
df['IsRemodeled'] = (df['YearRemodAdd'] > df['YearBuilt']).astype(int)

# (Ratio of above-grade living area to the total lot area)
# Usefulness: This ratio provides insight into land usage. A higher ratio means a larger house relative to the yard, which could be positive or negative depending on buyer preferences and neighborhood context. It helps the model understand the balance between indoor and outdoor space.
# Input samples: {'GrLivArea': [1144, 2520, 1520], 'LotArea': [10000, 14541, 4500]}
df['LivArea_LotArea_Ratio'] = df['GrLivArea'] / (df['LotArea'] + 1e-6)

# (Ratio of bathrooms to total rooms above grade)
# Usefulness: This feature can indicate the convenience and modernity of a house's layout. A higher ratio of bathrooms to rooms is often found in more modern or higher-end homes, which can be a positive price indicator.
# Input samples: {'TotalBath': [2.0, 4.0, 1.0], 'TotRmsAbvGrd': [6, 10, 6]}
df['Bath_per_Room'] = df['TotalBath'] / (df['TotRmsAbvGrd'] + 1e-6)

# (Total number of rooms including basement)
# Usefulness: While TotRmsAbvGrd is useful, it ignores finished basement space. This feature attempts to create a more holistic room count. Assuming a finished basement adds roughly 2-3 rooms on average. This is an approximation but can capture the value of a finished basement more effectively.
# Input samples: {'TotRmsAbvGrd': [6, 10, 6], 'BsmtFinSF1': [594, 1012, 182], 'BsmtFinSF2': [0, 0, 0]}
df['TotalRms'] = df['TotRmsAbvGrd'] + (df['BsmtFinSF1'] > 0).astype(int) + (df['BsmtFinSF2'] > 0).astype(int)

"""
    ]
}

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    split = pipeline.skb.train_test_split(random_state=seed, test_size=0.5)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    env_fit = split["train"]
    env_fit["sempipes_prefitted_state__house_features"] = state

    learner.fit(env_fit)

    env_eval = split["test"]
    env_eval["sempipes_prefitted_state__house_features"] = state

    y_pred = learner.predict(env_eval)

    score = rmsle(env_eval["_skrub_y"], y_pred)
    print(f"RMSLE on split {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
