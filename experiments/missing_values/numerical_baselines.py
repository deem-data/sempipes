import itertools
import json
import os
import ssl
import numpy as np
import pandas as pd
import skrub
from tqdm import tqdm

import sempipes

# Fix SSL certificate verification for dataset downloads
# This is needed on some systems where certificates aren't properly configured
ssl._create_default_https_context = ssl._create_unverified_context

# Enable experimental IterativeImputer
import warnings

from fancyimpute import KNN, IterativeImputer, MatrixFactorization, SimpleFill
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

np.random.seed(0)
DIR_PATH = "experiments/missing_values/"

# this appears to be neccessary for not running into too many open files errors
import resource

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))


def dict_product(hp_dict):
    """
    Returns cartesian product of hyperparameters
    """
    return [dict(zip(hp_dict.keys(), vals)) for vals in itertools.product(*hp_dict.values())]


def evaluate_mse(X_imputed, X, mask):
    return ((X_imputed[mask] - X[mask]) ** 2).mean()


def fancyimpute_hpo(fancyimputer, param_candidates, X, mask, percent_validation=10):
    # first generate all parameter candidates for grid search
    all_param_candidates = dict_product(param_candidates)
    # get linear indices of all training data points
    train_idx = (mask.reshape(np.prod(X.shape)) == False).nonzero()[0]
    # get the validation mask
    n_validation = int(len(train_idx) * percent_validation / 100)
    validation_idx = np.random.choice(train_idx, n_validation)
    validation_mask = np.zeros(np.prod(X.shape))
    validation_mask[validation_idx] = 1
    validation_mask = validation_mask.reshape(X.shape) > 0
    # save the original data
    X_incomplete = X.copy()
    # set validation and test data to nan
    X_incomplete[mask | validation_mask] = np.nan
    mse_hpo = []
    for params in all_param_candidates:
        if fancyimputer.__name__ != "SimpleFill":
            params["verbose"] = False
        X_imputed = fancyimputer(**params).fit_transform(X_incomplete)
        mse = evaluate_mse(X_imputed, X, validation_mask)
        print(f"Trained {fancyimputer.__name__} with {params}, mse={mse}")
        mse_hpo.append(mse)

    best_params = all_param_candidates[np.array(mse_hpo).argmin()]
    # now retrain with best params on all training data
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    X_imputed = fancyimputer(**best_params).fit_transform(X_incomplete)
    mse_best = evaluate_mse(X_imputed, X, mask)
    print(f"HPO: {fancyimputer.__name__}, best {best_params}, mse={mse_best}")
    return mse_best


def impute_mean(X, mask):
    return fancyimpute_hpo(SimpleFill, {"fill_method": ["mean"]}, X, mask)


def impute_knn(X, mask, hyperparams={"k": [2, 4, 6]}):
    return fancyimpute_hpo(KNN, hyperparams, X, mask)


def impute_mf(X, mask, hyperparams={"rank": [5, 10, 50], "shrinkage_value": [0, 1e-3, 1e-5]}):
    return fancyimpute_hpo(MatrixFactorization, hyperparams, X, mask)


def impute_sklearn_rf(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    reg = RandomForestRegressor(random_state=0)
    parameters = {"n_estimators": [2, 10, 100], "max_features": [int(np.sqrt(X.shape[-1])), X.shape[-1]]}
    clf = GridSearchCV(reg, parameters, cv=5)
    X_pred = IterativeImputer(random_state=0, estimator=reg).fit_transform(X_incomplete)
    mse = evaluate_mse(X_pred, X, mask)
    return mse


def impute_sklearn_linreg(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    reg = LinearRegression()
    X_pred = IterativeImputer(random_state=0, estimator=reg).fit_transform(X_incomplete)
    mse = evaluate_mse(X_pred, X, mask)
    return mse


def impute_sklearn_mice(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        initial_strategy="mean",
        max_iter=50,
        imputation_order="ascending",
        n_nearest_features=None,
        skip_complete=True,
        random_state=0,
    )
    X_pred = imputer.fit_transform(X_incomplete)

    mse = evaluate_mse(X_pred, X, mask)
    return mse


def impute_sempipes(X, mask, dataset):
    columns_to_impute = X.iloc[:, mask.sum(axis=0) > 0].columns.tolist()

    nl_prompts_fillna = {
        "load_diabetes": {
            "age": "Infer the age in years from the relevant diabetes-related attributes, especially metabolic and blood pressure measurements which may correlate with age.",
            "bmi": "Infer the body mass index from the relevant diabetes-related attributes, particularly blood pressure and metabolic measurements which often correlate with BMI.",
            "bp": "Infer the average blood pressure from the relevant diabetes-related attributes, especially age and BMI which are known to influence blood pressure.",
            "blood_sugar_level": "Infer the blood sugar level from the relevant diabetes-related attributes, especially BMI and other metabolic measurements which strongly correlate with blood sugar.",
            "total_serum_cholesterol": "Infer the total serum cholesterol from the relevant diabetes-related attributes, especially other cholesterol measurements (HDL, LDL) which are directly related.",
            "total_cholesterol_HDL": "Infer the total cholesterol / HDL ratio from the relevant diabetes-related attributes, especially total cholesterol and blood sugar measurements which directly determine this ratio.",
        },
        "load_wine": {
            "alcohol": "Infer the alcohol content from relevant wine-related attributes, especially proline and phenols content which indicate wine quality and correlate with alcohol levels. Consider that low alcohol and low proline wines are usually of a worse quality and producrs might want to hide this. The datasetis small, so consider models that perform well on small amounts of data",
            "flavanoids": "Infer the flavanoids content from relevant wine-related attributes, especially total phenols which include flavanoids as a key component. The datasetis small, so consider models that perform well on small amounts of data",
            "color_intensity": "Infer the color intensity content from relevant wine-related attributes, especially flavanoids and phenolic compounds which contribute to wine color. The datasetis small, so consider models that perform well on small amounts of data",
            "proline": "Infer the proline content from relevant wine-related attributes, especially alcohol content which correlates with wine quality indicators like proline. The datasetis small, so consider models that perform well on small amounts of data",
        },
        "fetch_california_housing": {
            "MedInc": "Infer the median income from relevant California housing-related attributes, especially average number of rooms/bedrooms, and house age which correlate with income levels.",
            "HouseAge": "Infer the house age from relevant California housing-related attributes, especially average number of rooms/bedrooms, and median income which correlate with house age.",
            "AveRooms": "Infer the average number of rooms from relevant California housing-related attributes, especially house age, population, occupancy and median income which relate to house size.",
            "Population": "Infer the area population from relevant California housing-related attributes, especially average number of rooms/bedrooms, and house age.",
            "AveBedrms": "Infer the average number of bedrooms from relevant California housing-related attributes, especially average number of rooms which relates to how many people can occupy a house.",
            "AveOccup": "Infer the average occupation (occupancy) from relevant California housing-related attributes, especially average number of rooms which relates to how many people can occupy a house.",
        },
        "load_breast_cancer": {
            "mean radius": "Infer the mean radius from relevant breast cancer-related attributes, especially pay attention to other features related to radius and perimiter.",
            "mean perimeter": "Infer the mean perimeter from relevant breast cancer-related attributes, especially radius- and symmetry-related attributes.",
            "mean smoothness": "Infer the mean smoothness from relevant breast cancer-related attributes. Take into account smoothness- and texture-related attributes.",
            "mean texture": "Infer the mean texture from relevant breast cancer-related attributes. Take into account smoothness- and texture-related attributes.",
        },
    }

    nl_prompts_clean = {
        "load_diabetes": "Impute missing values in the diabetes dataset. Missing values are marked with NaN. First, check if there are any missing values. Values are scaled by mean and standard deviation. Consider relationships between related measurements (e.g., cholesterol components, metabolic indicators, BMI). Use statistical methods like MICE, Random Forests, or Iterative Imputer that can capture these relationships. Pay attention to extreme values which may require special handling.",
        "load_wine": "Impute missing values and outliers in the wine dataset, check if some values are wrong or incorrect. First, check if there are any missing values. Consider relationships between wine quality indicators (e.g., phenols and flavanoids, alcohol, and proline). Note that some quality-related measurements may be correlated. Consider that low alcohol and low proline wines are usually of a worse quality and producers might want to hide this.",
        "fetch_california_housing": "Impute missing values in the California housing dataset. Missing values are marked with NaN. First, check if there are any missing values. Consider relationships between housing features (e.g., rooms and occupancy, income and house characteristics). Use statistical methods like MICE, Random Forests, or Iterative Imputer. Be aware that extreme values in certain features (like very large houses or very high income) may have different patterns.",
        "load_breast_cancer": "Impute missing values in the breast cancer dataset. Missing values are marked with NaN. First, check if there are any missing values. Consider relationships between tumor measurements (e.g., radius and perimeter, smoothness measurements). Use statistical methods like MICE, Random Forests, or Iterative Imputer that can capture geometric relationships between measurements.",
    }

    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan

    X_incomplete_var = skrub.var("X_dataset", X_incomplete)

    if dataset in nl_prompts_fillna:
        for column in columns_to_impute:
            X_incomplete_var = X_incomplete_var.sem_fillna(
                target_column=column,
                nl_prompt=nl_prompts_fillna[dataset][column],
                impute_with_existing_values_only=True,
                with_llm_only=False,
            )

    X_incomplete_var = X_incomplete_var.sem_clean(
        nl_prompt=nl_prompts_clean[dataset],
        columns=columns_to_impute,
    )

    X_pred = X_incomplete_var.skb.eval().values

    mse = evaluate_mse(X_pred, X.values, mask)
    return mse


def get_data(data_fn):
    if data_fn.__name__ == "load_diabetes":
        X = data_fn(as_frame=True, scaled=True)["data"]
        X.rename(
            columns={
                "s1": "total_serum_cholesterol",
                "s2": "low_density_lipoproteins",
                "s3": "high_density_lipoproteins",
                "s4": "total_cholesterol_HDL",
                "s5": "log_serum_triglycerides",
                "s6": "blood_sugar_level",
            },
            inplace=True,
        )
    # elif data_fn.__name__ == "load_wine":
    #     X = data_fn(as_frame=True, scaled=True)
    else:
        X, _ = data_fn(return_X_y=True, as_frame=True)
        # print(X.head())
    return X


def generate_missing_mask_by_dataset(X_df, dataset_name, percent_missing=30, missingness="MCAR"):
    """
    Generate missing value mask based on dataset-specific patterns.
    
    Args:
        X_df: DataFrame with the data
        dataset_name: Name of the dataset function (e.g., "load_diabetes")
        percent_missing: Percentage of values to make missing
        missingness: Type of missingness ("MCAR", "MAR", "MNAR")
    
    Returns:
        Boolean mask array of same shape as X_df
    """
    X = X_df.values
    mask = np.zeros(X.shape, dtype=bool)
    n_samples = X.shape[0]
    n_values_to_discard = int((percent_missing / 100) * n_samples)
    
    # Define columns to mask for each dataset
    columns_to_mask = {
        "load_diabetes": {
            "MCAR": ["bmi", "blood_sugar_level"],
            "MAR": {
                "total_serum_cholesterol": "total_cholesterol_HDL",  # LDL/HDL cholesterol missing when total cholesterol is low/normal. 
                "age": "bp",  # Younger patients are less likely to have blood pressure recorded in a clinic dataset.
            },
            "MNAR": ["age", "bmi", "blood_sugar_level"],  # Missing for extreme values (very high blood sugar or very high BMI)
        },
        "load_wine": {
            "MCAR": ["color_intensity", "flavanoids"],
            "MAR": {
                "total_phenols": "flavanoids",  # Flavanoids missing when total phenols is low
                "proline": "alcohol",  # Alcohol missing when proline is low (lower quality wines)
            },
            "MNAR": ["alcohol", "proline"],  # Missing for very low alcohol and low proline
        },
        
        "load_breast_cancer": {
            "MCAR": ["mean radius", "mean texture"],
            "MAR": 
            {
                "mean perimeter": "mean radius", # if the mean perimeter is small, mean radius is missing
                "smoothness error": "mean smoothness", # if the smoothness error is large, mean smoothness is missing
            },
            "MNAR": ["mean radius", "mean perimeter"],  # Missing for large tumors
        },

        "fetch_california_housing": {
            "MCAR": ["HouseAge", "AveBedrms"],
            "MAR": {
                "AveRooms": "AveOccup", # if the number of rooms is high, hide low occupation
                "AveRooms": "MedInc", # if the number of rooms is high, hide high income
            }, 
            "MNAR": ["MedInc", "AveRooms"],  # Missing for very expensive houses and big houses
        },
    }
    
    # Get column indices to mask
    target_cols = columns_to_mask[dataset_name][missingness]
    col_indices = [X_df.columns.get_loc(col) for col in target_cols]
    
    if missingness == "MCAR":
        # Missing completely at random - random missingness in target columns only
        for col_idx in col_indices:
            col_mask = np.random.rand(n_samples) < (percent_missing / 100.0)
            mask[:, col_idx] = col_mask
    
    elif missingness == "MAR":
        # Missing at random - missingness depends on another observed column
        # Get MAR configuration (dict of {column_to_mask: condition_column})
        mar_config = columns_to_mask[dataset_name]["MAR"]
        
        for condition_col_name, col_to_mask in mar_config.items():
            col_idx = X_df.columns.get_loc(col_to_mask)
            condition_col_idx = X_df.columns.get_loc(condition_col_name)
            condition_values = X[:, condition_col_idx]
            
            # Calculate number of values to discard for this column
            n_values_to_discard = int((percent_missing / 100) * n_samples)
            
            # Determine missingness direction based on dataset and column pair
            # Default: missing when condition is low (bottom percentile)
            if dataset_name == "load_diabetes":
                if col_to_mask == "total_cholesterol_HDL":
                    # Missing when total_serum_cholesterol is low/normal
                    threshold = np.percentile(condition_values, percent_missing + 5)
                    candidate_mask = condition_values < threshold
                elif col_to_mask == "bp":
                    # Missing when age is low (younger patients)
                    threshold = np.percentile(condition_values, percent_missing + 5)
                    candidate_mask = condition_values < threshold
            
            elif dataset_name == "load_wine":
                if col_to_mask == "flavanoids":
                    # Missing when total_phenols is low
                    threshold = np.percentile(condition_values, percent_missing + 5)
                    candidate_mask = condition_values < threshold
                elif col_to_mask == "alcohol":
                    # Missing when proline is low (lower quality wines)
                    threshold = np.percentile(condition_values, percent_missing + 5)
                    candidate_mask = condition_values < threshold
                    
            elif dataset_name == "load_breast_cancer":
                if col_to_mask == "mean radius":
                    # Missing when mean perimeter is small
                    threshold = np.percentile(condition_values, percent_missing + 5)
                    candidate_mask = condition_values < threshold
                elif col_to_mask == "mean smoothness":
                    # Missing when smoothness error is large (top percentile)
                    threshold = np.percentile(condition_values, 100 - percent_missing - 5)
                    candidate_mask = condition_values > threshold
                    
            elif dataset_name == "fetch_california_housing":
                # Condition column is AveRooms (when it's too low or too high)
                # Check if AveRooms is at extremes (bottom or top percentile)
                lower_threshold = np.percentile(condition_values, (percent_missing + 5)/ 2)
                upper_threshold = np.percentile(condition_values, 100 - (percent_missing + 5) / 2)
                rows_with_extreme_rooms = (condition_values < lower_threshold) | (condition_values > upper_threshold)
                
                if col_to_mask == "AveOccup":
                    # When AveRooms is high, hide AveOccup 
                    candidate_mask = condition_values > upper_threshold
                elif col_to_mask == "MedInc":
                    # When AveRooms is extreme, hide MedInc
                    candidate_mask = rows_with_extreme_rooms
            
            # Randomly select n_values_to_discard from candidates
            candidate_indices = np.where(candidate_mask)[0]
            if len(candidate_indices) > 0:
                if len(candidate_indices) <= n_values_to_discard:
                    # If we have fewer candidates than needed, mask all of them
                    selected_indices = candidate_indices
                    print(f"WARNING: No candidates found for {col_to_mask} in {dataset_name} with {missingness} missingness.")
                else:
                    # Randomly select the required number
                    selected_indices = np.random.choice(candidate_indices, size=n_values_to_discard, replace=False)
                
                col_mask = np.zeros(n_samples, dtype=bool)
                col_mask[selected_indices] = True
            else:
                # No candidates found, use random fallback
                col_mask = np.random.rand(n_samples) < (percent_missing / 100.0)
            
            mask[:, col_idx] = col_mask
    
    elif missingness == "MNAR":
        # Missing not at random - missingness depends on the value itself
        for col_idx in col_indices:
            col_values = X[:, col_idx]
            
            # Calculate number of values to discard for this column
            n_values_to_discard = int((percent_missing / 100) * n_samples)
            
            # Determine candidate mask based on dataset and column
            if dataset_name == "fetch_california_housing":
                if "MedInc" in target_cols and X_df.columns[col_idx] == "MedInc":
                    # Income missing for very high income (people might not report)
                    threshold = np.percentile(col_values, 100 - percent_missing - 5)
                    candidate_mask = col_values > threshold
                elif "AveRooms" in target_cols and X_df.columns[col_idx] == "AveRooms":
                    # Rooms missing for very large houses (luxury properties)
                    threshold = np.percentile(col_values, 100 - percent_missing - 5)
                    candidate_mask = col_values > threshold
            elif dataset_name == "load_diabetes":
                if "bmi" in target_cols and X_df.columns[col_idx] == "bmi":
                    # BMI missing for extreme values (very high or very low)
                    lower_threshold = np.percentile(col_values, (percent_missing + 5) / 2)
                    upper_threshold = np.percentile(col_values, 100 - (percent_missing + 5) / 2)
                    candidate_mask = (col_values < lower_threshold) | (col_values > upper_threshold)
                elif "blood_sugar_level" in target_cols and X_df.columns[col_idx] == "blood_sugar_level":
                    # Blood sugar missing for very high values (diabetic patients)
                    threshold = np.percentile(col_values, 100 - percent_missing - 5)
                    candidate_mask = col_values > threshold
                elif "age" in target_cols and X_df.columns[col_idx] == "age":
                    # Age missing for extreme values
                    lower_threshold = np.percentile(col_values, (percent_missing + 5) / 2)
                    upper_threshold = np.percentile(col_values, 100 - (percent_missing + 5) / 2)
                    candidate_mask = (col_values < lower_threshold) | (col_values > upper_threshold)
            elif dataset_name == "load_wine":
                if "alcohol" in target_cols and X_df.columns[col_idx] == "alcohol":
                    # Alcohol missing for very low alcohol content (lower quality wines)
                    threshold = np.percentile(col_values, percent_missing + 5)
                    candidate_mask = col_values < threshold
                elif "proline" in target_cols and X_df.columns[col_idx] == "proline":
                    # Proline missing for very low values (lower quality wines)
                    threshold = np.percentile(col_values, percent_missing + 5)
                    candidate_mask = col_values < threshold
            elif dataset_name == "load_breast_cancer":
                if "mean radius" in target_cols and X_df.columns[col_idx] == "mean radius":
                    # Missing for large tumors
                    threshold = np.percentile(col_values, 100 - percent_missing - 5)
                    candidate_mask = col_values > threshold
                elif "mean perimeter" in target_cols and X_df.columns[col_idx] == "mean perimeter":
                    # Missing for large tumors
                    threshold = np.percentile(col_values, 100 - percent_missing - 5)
                    candidate_mask = col_values > threshold
            else:
                # Default: missing for extreme high values
                threshold = np.percentile(col_values, 100 - percent_missing - 5)
                candidate_mask = col_values > threshold
            
            # Randomly select n_values_to_discard from candidates
            candidate_indices = np.where(candidate_mask)[0]
            if len(candidate_indices) > 0:
                if len(candidate_indices) <= n_values_to_discard:
                    # If we have fewer candidates than needed, mask all of them
                    print(f"WARNING: No candidates found for {col_idx} in {dataset_name} with {missingness} missingness.")
                    selected_indices = candidate_indices
                else:
                    # Randomly select the required number
                    selected_indices = np.random.choice(candidate_indices, size=n_values_to_discard, replace=False)
                
                col_mask = np.zeros(n_samples, dtype=bool)
                col_mask[selected_indices] = True
            else:
                # No candidates found, use random fallback
                col_mask = np.random.rand(n_samples) < (percent_missing / 100.0)
            
            mask[:, col_idx] = col_mask
    
    return mask


def experiment(percent_missing_list=[30], nreps=5):
    DATA_LOADERS = [
        # load_wine,
        # fetch_california_housing,
        load_breast_cancer,
        # load_diabetes,
    ]

    imputers = [
        impute_sempipes,
        impute_mean,
        impute_knn,
        impute_mf,
        impute_sklearn_rf,
        impute_sklearn_linreg,
        impute_sklearn_mice,
    ]

    results = []
    patterns = ["MAR", "MCAR", "MNAR"]
    with open(os.path.join(DIR_PATH, "cancer_5runs_gpt5.json"), "w") as fh:
        for percent_missing in tqdm(percent_missing_list):
            for data_fn in DATA_LOADERS:
                X_df = get_data(data_fn)
                X = X_df.values
                for missingness in patterns:
                # for missingness in ["MNAR"]:
                    for _ in range(nreps):
                        missing_mask = generate_missing_mask_by_dataset(X_df, data_fn.__name__, percent_missing, missingness)
                        for imputer_fn in imputers:
                            print(imputer_fn.__name__)
                            mse = (
                                imputer_fn(X, missing_mask)
                                if imputer_fn.__name__ != "impute_sempipes"
                                else imputer_fn(X_df, missing_mask, dataset=data_fn.__name__)
                            )
                            result = {
                                "data": data_fn.__name__,
                                "imputer": imputer_fn.__name__,
                                "percent_missing": percent_missing,
                                "missingness": missingness,
                                "mse": mse,
                            }
                            fh.write(json.dumps(result) + "\n")

sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="openai/gpt-5",
            # parameters={"temperature": 0.0},
        )
    )
    
experiment()
