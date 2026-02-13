import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Load the data
all_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
seed = 42

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

for fold_index, (train_idx, test_idx) in enumerate(kf.split(all_data)):
    data = all_data.copy(deep=True)

    ###
    # Drop Id column if present
    if "Id" in data.columns:
        data = data.drop("Id", axis=1)

    # Separate target
    y = data["SalePrice"]
    X = data.drop("SalePrice", axis=1)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Log-transform target for evaluation
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    # train = all_data.iloc[train_idx]
    # test = all_data.iloc[test_idx]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    # Build pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))]
    )

    # Fit model
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    score = math.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))
    print(f"RMSLE on split {fold_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
