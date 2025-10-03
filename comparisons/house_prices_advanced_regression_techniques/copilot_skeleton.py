import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_error(y, y_predicted))


all_data_initial = pd.read_csv("comparisons/house_prices_advanced_regression_techniques/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    df = all_data_initial.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

    df_train, df_test = train_test_split(df, test_size=0.5, random_state=seed)

    y_train = df_train["SalePrice"]
    x_train = df_train.drop(["SalePrice"], axis=1)

    y_test = df_test["SalePrice"]
    df_test = df_test.drop(["SalePrice"], axis=1)

    # TODO add feature engineering code

    # TODO add model training and prediction

    y_pred = ...  # TODO Replace with actual predictions

    score = rmsle(y_test, y_pred)
    print(f"RMSLE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
