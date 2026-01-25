import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data
all_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
seed = 42
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_index, (train_idx, test_idx) in enumerate(kf.split(all_data)):
    train = all_data.iloc[train_idx]
    test = all_data.iloc[test_idx]

    # Identify key features for interaction terms based on domain knowledge
    key_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"]

    # Create interaction terms for both train and test datasets
    for i in range(len(key_features)):  # pylint: disable=consider-using-enumerate
        for j in range(i + 1, len(key_features)):
            name = key_features[i] + "_X_" + key_features[j]
            train[name] = train[key_features[i]] * train[key_features[j]]
            test[name] = test[key_features[i]] * test[key_features[j]]

    # Separate features and target variable
    X = train.drop(["SalePrice", "Id"], axis=1)
    y = np.log(train["SalePrice"])  # Log transformation
    test_ids = test["Id"]
    test = test.drop(["Id"], axis=1)
    y_test = np.log(test["SalePrice"])

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, X.select_dtypes(exclude=["object"]).columns),
            ("cat", categorical_transformer, X.select_dtypes(include=["object"]).columns),
        ]
    )

    # Define model
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Lasso(alpha=0.001))])

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on validation set
    preds_valid = model.predict(X_valid)

    # Evaluate the model
    # score = mean_squared_error(y_valid, preds_valid, squared=False)
    # print(f"Validation RMSE: {score}")

    # Predict on test data
    test_preds = model.predict(test)

    score = math.sqrt(mean_squared_error(y_test, test_preds))
    print(f"RMSLE on split {fold_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
