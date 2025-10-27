import random
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

all_data_initial = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")


def rmsle(y, y_predicted):
    return np.sqrt(mean_squared_log_error(y, y_predicted))


scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    random.seed(seed)

    df_train, df_test = train_test_split(all_data_initial, test_size=0.5, random_state=seed)

    # load datasets
    # df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
    # df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
    # df_sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

    # mark train and test sets for future split
    df_train["train_test"] = "Train"
    df_test["train_test"] = "Test"

    # combine to a single dataframe with all data for feature engineering
    df_all = pd.concat([df_train, df_test])

    # print dataset shape and columns
    # trow, tcol = df_train.shape
    # erow, ecol = df_test.shape
    # srow, scol = df_sample_submission.shape

    # drop the Id and PoolQC columns
    df_all = df_all.drop(["Id", "PoolQC", "PoolArea"], axis=1)

    # drop features with little information based on visualizations
    df_all = df_all.drop(
        [
            "BsmtFinSF2",
            "LowQualFinSF",
            "BsmtHalfBath",
            "KitchenAbvGr",
            "EnclosedPorch",
            "3SsnPorch",
            "MiscVal",
            "Street",
            "Utilities",
            "Condition2",
            "RoofMatl",
            "Heating",
            "MiscFeature",
        ],
        axis=1,
    )

    # drop features with little information based on heatmap
    df_all = df_all.drop(["MSSubClass", "OverallCond", "ScreenPorch", "MoSold", "YrSold"], axis=1)

    # replace numerical features with the mean of the column
    for col in df_all.columns:
        if (df_all[col].dtype == "float64") or (df_all[col].dtype == "int64"):
            df_all[col].fillna(df_train[col].mean(), inplace=True)

    # replace categorical features with the most common value of the column
    for col in df_all.columns:
        if df_all[col].dtype == "object":
            df_all[col].fillna(df_train[col].mode()[0], inplace=True)

    # encode ordinal features
    for col in ["BsmtQual", "BsmtCond"]:
        OE = OrdinalEncoder(categories=[["No", "Po", "Fa", "TA", "Gd", "Ex"]])
        df_all[col] = OE.fit_transform(df_all[[col]])

    for col in ["ExterQual", "ExterCond", "KitchenQual"]:
        OE = OrdinalEncoder(categories=[["Po", "Fa", "TA", "Gd", "Ex"]])
        df_all[col] = OE.fit_transform(df_all[[col]])

    OE = OrdinalEncoder(categories=[["N", "P", "Y"]])
    df_all["PavedDrive"] = OE.fit_transform(df_all[["PavedDrive"]])

    OE = OrdinalEncoder(categories=[["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"]])
    df_all["Electrical"] = OE.fit_transform(df_all[["Electrical"]])

    for col in ["BsmtFinType1", "BsmtFinType2"]:
        OE = OrdinalEncoder(categories=[["No", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]])
        df_all[col] = OE.fit_transform(df_all[[col]])

    OE = OrdinalEncoder(categories=[["C (all)", "RH", "RM", "RL", "FV"]])
    df_all["MSZoning"] = OE.fit_transform(df_all[["MSZoning"]])

    OE = OrdinalEncoder(categories=[["Slab", "BrkTil", "Stone", "CBlock", "Wood", "PConc"]])
    df_all["Foundation"] = OE.fit_transform(df_all[["Foundation"]])

    OE = OrdinalEncoder(
        categories=[
            [
                "MeadowV",
                "IDOTRR",
                "BrDale",
                "Edwards",
                "BrkSide",
                "OldTown",
                "NAmes",
                "Sawyer",
                "Mitchel",
                "NPkVill",
                "SWISU",
                "Blueste",
                "SawyerW",
                "NWAmes",
                "Gilbert",
                "Blmngtn",
                "ClearCr",
                "Crawfor",
                "CollgCr",
                "Veenker",
                "Timber",
                "Somerst",
                "NoRidge",
                "StoneBr",
                "NridgHt",
            ]
        ]
    )
    df_all["Neighborhood"] = OE.fit_transform(df_all[["Neighborhood"]])

    OE = OrdinalEncoder(categories=[["None", "BrkCmn", "BrkFace", "Stone"]])
    df_all["MasVnrType"] = OE.fit_transform(df_all[["MasVnrType"]])

    OE = OrdinalEncoder(categories=[["AdjLand", "Abnorml", "Alloca", "Family", "Normal", "Partial"]])
    df_all["SaleCondition"] = OE.fit_transform(df_all[["SaleCondition"]])

    OE = OrdinalEncoder(categories=[["Gambrel", "Gable", "Hip", "Mansard", "Flat", "Shed"]])
    df_all["RoofStyle"] = OE.fit_transform(df_all[["RoofStyle"]])

    # scale all numerical features
    numerical_features = df_all.select_dtypes(exclude="object").columns

    scaler = StandardScaler()
    df_train_for_scaling = df_all[df_all["train_test"] == "Train"]
    scaler.fit(df_train_for_scaling[numerical_features])
    df_all[numerical_features] = scaler.transform(df_all[numerical_features])

    # re add SalePrice
    # df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
    # df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
    # df_all2 = pd.concat([df_train, df_test])

    # df_all['SalePrice'] = df_all2['SalePrice']

    # ONE HOT ENCODING - COMING SOON

    # one_hot_features = ['Alley',
    #                    'LotShape',
    #                    'LandContour',
    #                    'LotConfig',
    #                    'LandSlope',
    #                    'Condition1',
    #                    'GarageQual',
    #                    'GarageCond',
    #                    'Fence',
    #                    'SaleType']

    # df_dummies = pd.get_dummies(df_all[one_hot_features])

    # df_all = df_all.drop(one_hot_features, axis=1)

    # df_all = df_all.join(df_dummies)

    # df_all

    drop_col = [
        "Alley",
        "LotShape",
        "LandContour",
        "LotConfig",
        "LandSlope",
        "Condition1",
        "GarageQual",
        "GarageCond",
        "Fence",
        "SaleType",
        "BldgType",
        "HouseStyle",
        "Exterior1st",
        "Exterior2nd",
        "GarageFinish",
        "GarageType",
        "FireplaceQu",
        "Functional",
        "BsmtExposure",
        "HeatingQC",
        "CentralAir",
    ]

    df_all = df_all.drop(drop_col, axis=1)

    # resplit into train and test sets
    X_train = df_all[df_all["train_test"] == "Train"].drop(["train_test", "SalePrice"], axis=1)
    X_test = df_all[df_all["train_test"] == "Test"].drop(["train_test", "SalePrice"], axis=1)
    y_train = df_all[df_all["train_test"] == "Train"]["SalePrice"]
    y_test = df_all[df_all["train_test"] == "Test"]["SalePrice"]

    # Set all values in y_train and y_test to at least 0
    y_train = y_train.apply(lambda x: max(0, x))
    y_test = y_test.apply(lambda x: max(0, x))

    # instanciate a XGBoost Regressor model
    xgb = XGBRegressor(random_state=42)

    # set up a parameter grid to search for the best combination of hyperparameters
    parameter_grid = {
        "n_estimators": [500, 525, 550],
        "max_depth": [3],
        "learning_rate": [0.04, 0.05],
        "colsample_bytree": [0.25, 0.3],
        "subsample": [0.65, 0.7, 0.75],
    }

    # fit the model with all combinations of the parameters from the grid
    xgb_grid = GridSearchCV(estimator=xgb, param_grid=parameter_grid, cv=3, verbose=False, n_jobs=-1)

    best_xgb_model = xgb_grid.fit(X_train, y_train)

    # define simple function to judge model performance
    # def model_performance(model, name):
    #    print(name)
    #    print(f'Best Score: {model.best_score_}')
    #    print(f'Best Parameters: {model.best_params_}\n')

    # evaluate the model
    # model_performance(best_xgb_model, 'XGBoost Regressor (GridSearchCV)')

    best_xgb_model.fit(X_train, y_train)

    # use the model to predict on the test set
    y_pred = best_xgb_model.predict(X_test).astype(int)
    score = rmsle(y_test, y_pred)
    print(f"RMSLE on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
