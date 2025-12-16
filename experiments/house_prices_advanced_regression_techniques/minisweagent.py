import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load the data
all_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):

    data = all_data.copy(deep=True)

    ###
    # Drop Id column if present
    if 'Id' in data.columns:
        data = data.drop('Id', axis=1)

    # Separate target
    y = data['SalePrice']
    X = data.drop('SalePrice', axis=1)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Log-transform target for evaluation
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    # Build pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Fit model
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)


    # train, test = train_test_split(all_data, test_size=0.5, random_state=seed)

    # # Identify key features for interaction terms based on domain knowledge
    # key_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"]

    # # Create interaction terms for both train and test datasets
    # for i in range(len(key_features)):  # pylint: disable=consider-using-enumerate
    #     for j in range(i + 1, len(key_features)):
    #         name = key_features[i] + "_X_" + key_features[j]
    #         train[name] = train[key_features[i]] * train[key_features[j]]
    #         test[name] = test[key_features[i]] * test[key_features[j]]

    # # Separate features and target variable
    # X = train.drop(["SalePrice", "Id"], axis=1)
    # y = np.log(train["SalePrice"])  # Log transformation
    # test_ids = test["Id"]
    # test = test.drop(["Id"], axis=1)
    # y_test = np.log(test["SalePrice"])

    # # Preprocessing for numerical data
    # numerical_transformer = Pipeline(
    #     steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    # )

    # # Preprocessing for categorical data
    # categorical_transformer = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="most_frequent")),
    #         ("onehot", OneHotEncoder(handle_unknown="ignore")),
    #     ]
    # )

    # # Bundle preprocessing for numerical and categorical data
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", numerical_transformer, X.select_dtypes(exclude=["object"]).columns),
    #         ("cat", categorical_transformer, X.select_dtypes(include=["object"]).columns),
    #     ]
    # )

    # # Define model
    # model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Lasso(alpha=0.001))])

    # # Split data into training and validation sets
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # # Train the model
    # model.fit(X_train, y_train)

    # # Predict on validation set
    # preds_valid = model.predict(X_valid)

    # # Evaluate the model
    # # score = mean_squared_error(y_valid, preds_valid, squared=False)
    # # print(f"Validation RMSE: {score}")

    # Predict on test data
    #test_preds = model.predict(test)

    score = math.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))
    print(f"RMSLE on split {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
