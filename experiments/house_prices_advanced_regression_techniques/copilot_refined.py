import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_error(y, y_predicted))


all_data_initial = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    df = all_data_initial.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

    df_train, df_test = train_test_split(df, test_size=0.5, random_state=seed)

    y_train = df_train["SalePrice"]
    x_train = df_train.drop(["SalePrice"], axis=1)

    y_test = df_test["SalePrice"]
    df_test = df_test.drop(["SalePrice"], axis=1)

    # Feature engineering: impute missing values, encode categoricals, scale numerics
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns
    categorical_cols = x_train.select_dtypes(exclude=[np.number]).columns

    # Impute numerics
    num_imputer = SimpleImputer(strategy="median")
    x_train_num = num_imputer.fit_transform(x_train[numeric_cols])
    x_test_num = num_imputer.transform(df_test[numeric_cols])

    # Encode categoricals
    cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    x_train_cat = cat_imputer.fit_transform(x_train[categorical_cols])
    x_test_cat = cat_imputer.transform(df_test[categorical_cols])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train_cat_enc = encoder.fit_transform(x_train_cat)
    x_test_cat_enc = encoder.transform(x_test_cat)

    # Scale numerics
    scaler = StandardScaler()
    x_train_num_scaled = scaler.fit_transform(x_train_num)
    x_test_num_scaled = scaler.transform(x_test_num)

    # Concatenate features
    X_train_final = np.hstack([x_train_num_scaled, x_train_cat_enc])
    X_test_final = np.hstack([x_test_num_scaled, x_test_cat_enc])

    # Model training and prediction
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    score = rmsle(y_test, y_pred)
    print(f"RMSLE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
