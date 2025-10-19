# mypy: ignore-errors
# https://www.kaggle.com/code/kamalchhirang/eda-feature-engineering-lgb-xgb-cat/notebook#Feature-Engineering-&-Prediction
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")


def xgb_model(trn_x, trn_y, val_x, val_y, test, verbose):
    params = {
        "objective": "reg:squarederror",  # Updated from 'reg:linear'
        "eta": 0.01,
        "max_depth": 6,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "eval_metric": "rmse",
        "seed": random_seed,
        "verbosity": 0,  # Changed from 'silent': True
    }

    evals_result = {}
    model = xgb.train(
        params,
        xgb.DMatrix(trn_x, trn_y),
        100000,
        evals=[(xgb.DMatrix(trn_x, trn_y), "train"), (xgb.DMatrix(val_x, val_y), "valid")],
        verbose_eval=verbose,
        early_stopping_rounds=500,
        evals_result=evals_result,
    )
    best_idx = np.argmin(np.array(evals_result["valid"]["rmse"]))

    val_pred = model.predict(xgb.DMatrix(val_x), iteration_range=(0, model.best_iteration))
    test_pred = model.predict(xgb.DMatrix(test), iteration_range=(0, model.best_iteration))

    return {
        "val": val_pred,
        "test": test_pred,
        "error": evals_result["valid"]["rmse"][best_idx],
        "importance": [i for k, i in model.get_score().items()],
    }


def lgb_model(trn_x, trn_y, val_x, val_y, test, verbose):
    params = {
        "objective": "regression",
        "num_leaves": 30,
        "min_data_in_leaf": 20,
        "max_depth": 9,
        "learning_rate": 0.004,
        # 'min_child_samples':100,
        "feature_fraction": 0.9,
        "bagging_freq": 1,
        "bagging_fraction": 0.9,
        "lambda_l1": 0.2,
        "bagging_seed": random_seed,
        "metric": "rmse",
        # 'subsample':.8,
        # 'colsample_bytree':.9,
        "random_state": random_seed,
        "verbosity": -1,
    }

    record = dict()
    model = lgb.train(
        params,
        lgb.Dataset(trn_x, trn_y),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(val_x, val_y)],
        callbacks=[lgb.record_evaluation(record), lgb.log_evaluation(verbose)],
    )

    best_idx = np.argmin(np.array(record["valid_0"]["rmse"]))

    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    test_pred = model.predict(test, num_iteration=model.best_iteration)

    return {
        "val": val_pred,
        "test": test_pred,
        "error": record["valid_0"]["rmse"][best_idx],
        "importance": model.feature_importance("gain"),
    }


def cat_model(trn_x, trn_y, val_x, val_y, test, verbose):
    model = CatBoostRegressor(
        iterations=100000,
        learning_rate=0.004,
        depth=5,
        eval_metric="RMSE",
        colsample_bylevel=0.8,
        random_seed=random_seed,
        bagging_temperature=0.2,
        metric_period=None,
        early_stopping_rounds=200,
    )
    model.fit(trn_x, trn_y, eval_set=(val_x, val_y), use_best_model=True, verbose=False)

    val_pred = model.predict(val_x)
    test_pred = model.predict(test)

    # Get the best score - structure varies by CatBoost version
    best_scores = model.get_best_score()
    if "validation" in best_scores:
        error = best_scores["validation"]["RMSE"]
    elif "validation_0" in best_scores:
        error = best_scores["validation_0"]["RMSE"]
    else:
        # Fallback: calculate RMSE manually
        error = np.sqrt(np.mean((val_pred - val_y) ** 2))

    return {"val": val_pred, "test": test_pred, "error": error}


def get_json_dict(df):
    global json_cols
    result = dict()
    for e_col in json_cols:
        d = dict()
        rows = df[e_col].values
        for row in rows:
            if row is None:
                continue
            for i in row:
                if i["name"] not in d:
                    d[i["name"]] = 0
                d[i["name"]] += 1
        result[e_col] = d
    return result


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


# Rewrote this to only compute statistics on the training data to remove data leakage
def prepare(df, train_copy_for_stats):
    global json_cols
    global train_dict

    df[["release_month", "release_day", "release_year"]] = (
        df["release_date"].str.split("/", expand=True).replace(np.nan, 0).astype(int)
    )
    df["release_year"] = df["release_year"]
    df.loc[(df["release_year"] <= 19) & (df["release_year"] < 100), "release_year"] += 2000
    df.loc[(df["release_year"] > 19) & (df["release_year"] < 100), "release_year"] += 1900

    releaseDate = pd.to_datetime(df["release_date"])
    df["release_dayofweek"] = releaseDate.dt.dayofweek
    df["release_quarter"] = releaseDate.dt.quarter

    train_copy_for_stats[["release_month", "release_day", "release_year"]] = (
        train_copy_for_stats["release_date"].str.split("/", expand=True).replace(np.nan, 0).astype(int)
    )
    train_copy_for_stats["release_year"] = train_copy_for_stats["release_year"]
    train_copy_for_stats.loc[
        (train_copy_for_stats["release_year"] <= 19) & (train_copy_for_stats["release_year"] < 100), "release_year"
    ] += 2000
    train_copy_for_stats.loc[
        (train_copy_for_stats["release_year"] > 19) & (train_copy_for_stats["release_year"] < 100), "release_year"
    ] += 1900

    releaseDate = pd.to_datetime(train_copy_for_stats["release_date"])
    train_copy_for_stats["release_dayofweek"] = releaseDate.dt.dayofweek
    train_copy_for_stats["release_quarter"] = releaseDate.dt.quarter

    rating_na = train_copy_for_stats.groupby(["release_year", "original_language"])["rating"].mean().reset_index()
    rating_na.columns = ["release_year", "original_language", "rating_mean"]
    df = df.merge(rating_na, how="left", on=["release_year", "original_language"])
    df["rating"] = df["rating"].fillna(df["rating_mean"])
    df.drop("rating_mean", axis=1, inplace=True)

    vote_count_na = (
        train_copy_for_stats.groupby(["release_year", "original_language"])["totalVotes"].mean().reset_index()
    )
    vote_count_na.columns = ["release_year", "original_language", "totalVotes_mean"]
    df = df.merge(vote_count_na, how="left", on=["release_year", "original_language"])
    df["totalVotes"] = df["totalVotes"].fillna(df["totalVotes_mean"])
    df.drop("totalVotes_mean", axis=1, inplace=True)

    df["weightedRating"] = (df["rating"] * df["totalVotes"] + 6.367 * 1000) / (df["totalVotes"] + 1000)

    df["originalBudget"] = df["budget"]
    df["inflationBudget"] = df["budget"] + df["budget"] * 1.8 / 100 * (
        2018 - df["release_year"]
    )  # Inflation simple formula
    df["budget"] = np.log1p(df["budget"])

    # Thanks to this Kernel for the next 7 features https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
    df["genders_0_crew"] = df["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 0]))
    df["genders_1_crew"] = df["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 1]))
    df["genders_2_crew"] = df["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 2]))
    df["_collection_name"] = df["belongs_to_collection"].apply(lambda x: x[0]["name"] if x != {} else 0)
    le = LabelEncoder()
    le.fit(list(df["_collection_name"].fillna("")))
    df["_collection_name"] = le.transform(df["_collection_name"].fillna("").astype(str))
    df["_num_Keywords"] = df["Keywords"].apply(lambda x: len(x) if x != {} else 0)
    df["_num_cast"] = df["cast"].apply(lambda x: len(x) if x != {} else 0)

    df["_popularity_mean_year"] = train_copy_for_stats["popularity"] / train_copy_for_stats.groupby("release_year")[
        "popularity"
    ].transform("mean")
    df["_budget_runtime_ratio"] = df["budget"] / df["runtime"]
    df["_budget_popularity_ratio"] = df["budget"] / df["popularity"]
    df["_budget_year_ratio"] = df["budget"] / (df["release_year"] * df["release_year"])
    df["_releaseYear_popularity_ratio"] = df["release_year"] / df["popularity"]
    df["_releaseYear_popularity_ratio2"] = df["popularity"] / df["release_year"]

    df["_popularity_totalVotes_ratio"] = df["totalVotes"] / df["popularity"]
    df["_rating_popularity_ratio"] = df["rating"] / df["popularity"]
    df["_rating_totalVotes_ratio"] = df["totalVotes"] / df["rating"]
    df["_totalVotes_releaseYear_ratio"] = df["totalVotes"] / df["release_year"]
    df["_budget_rating_ratio"] = df["budget"] / df["rating"]
    df["_runtime_rating_ratio"] = df["runtime"] / df["rating"]
    df["_budget_totalVotes_ratio"] = df["budget"] / df["totalVotes"]

    df["has_homepage"] = 1
    df.loc[pd.isnull(df["homepage"]), "has_homepage"] = 0

    df["isbelongs_to_collectionNA"] = 0
    df.loc[pd.isnull(df["belongs_to_collection"]), "isbelongs_to_collectionNA"] = 1

    df["isTaglineNA"] = 0
    df.loc[df["tagline"] == 0, "isTaglineNA"] = 1

    df["isOriginalLanguageEng"] = 0
    df.loc[df["original_language"] == "en", "isOriginalLanguageEng"] = 1

    df["isTitleDifferent"] = 1
    df.loc[df["original_title"] == df["title"], "isTitleDifferent"] = 0

    df["isMovieReleased"] = 1
    df.loc[df["status"] != "Released", "isMovieReleased"] = 0

    # get collection id
    df["collection_id"] = df["belongs_to_collection"].apply(lambda x: np.nan if len(x) == 0 else x[0]["id"])

    df["original_title_letter_count"] = df["original_title"].str.len()
    df["original_title_word_count"] = df["original_title"].str.split().str.len()

    df["title_word_count"] = df["title"].str.split().str.len()
    df["overview_word_count"] = df["overview"].str.split().str.len()
    df["tagline_word_count"] = df["tagline"].str.split().str.len()

    df["production_countries_count"] = df["production_countries"].apply(lambda x: len(x))
    df["production_companies_count"] = df["production_companies"].apply(lambda x: len(x))
    df["cast_count"] = df["cast"].apply(lambda x: len(x))
    df["crew_count"] = df["crew"].apply(lambda x: len(x))

    df["meanruntimeByYear"] = train_copy_for_stats.groupby("release_year")["runtime"].aggregate("mean")
    df["meanPopularityByYear"] = train_copy_for_stats.groupby("release_year")["popularity"].aggregate("mean")
    df["meanBudgetByYear"] = train_copy_for_stats.groupby("release_year")["budget"].aggregate("mean")
    df["meantotalVotesByYear"] = train_copy_for_stats.groupby("release_year")["totalVotes"].aggregate("mean")
    df["meanTotalVotesByRating"] = train_copy_for_stats.groupby("rating")["totalVotes"].aggregate("mean")
    df["medianBudgetByYear"] = train_copy_for_stats.groupby("release_year")["budget"].aggregate("median")

    for col in ["genres", "production_countries", "spoken_languages", "production_companies"]:
        df[col] = (
            df[col]
            .map(
                lambda x: sorted(
                    list(set([n if n in train_dict[col] else col + "_etc" for n in [d["name"] for d in x]]))
                )
            )
            .map(lambda x: ",".join(map(str, x)))
        )
        temp = df[col].str.get_dummies(sep=",")
        df = pd.concat([df, temp], axis=1, sort=False)
    df.drop(["genres_etc"], axis=1, inplace=True)

    df = df.drop(
        [
            "id",
            "revenue",
            "belongs_to_collection",
            "genres",
            "homepage",
            "imdb_id",
            "overview",
            "runtime",
            "poster_path",
            "production_companies",
            "production_countries",
            "release_date",
            "spoken_languages",
            "status",
            "title",
            "Keywords",
            "cast",
            "crew",
            "original_language",
            "original_title",
            "tagline",
            "collection_id",
        ],
        axis=1,
    )

    df.fillna(value=0.0, inplace=True)

    return df


scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    train = pd.read_csv("comparisons/tmdb_box_office_prediction/data.csv")

    train.loc[train["id"] == 16, "revenue"] = 192864  # Skinning
    train.loc[train["id"] == 90, "budget"] = 30000000  # Sommersby
    train.loc[train["id"] == 118, "budget"] = 60000000  # Wild Hogs
    train.loc[train["id"] == 149, "budget"] = 18000000  # Beethoven
    train.loc[train["id"] == 313, "revenue"] = 12000000  # The Cookout
    train.loc[train["id"] == 451, "revenue"] = 12000000  # Chasing Liberty
    train.loc[train["id"] == 464, "budget"] = 20000000  # Parenthood
    train.loc[train["id"] == 470, "budget"] = 13000000  # The Karate Kid, Part II
    train.loc[train["id"] == 513, "budget"] = 930000  # From Prada to Nada
    train.loc[train["id"] == 797, "budget"] = 8000000  # Welcome to Dongmakgol
    train.loc[train["id"] == 819, "budget"] = 90000000  # Alvin and the Chipmunks: The Road Chip
    train.loc[train["id"] == 850, "budget"] = 90000000  # Modern Times
    train.loc[train["id"] == 1007, "budget"] = 2  # Zyzzyx Road
    train.loc[train["id"] == 1112, "budget"] = 7500000  # An Officer and a Gentleman
    train.loc[train["id"] == 1131, "budget"] = 4300000  # Smokey and the Bandit
    train.loc[train["id"] == 1359, "budget"] = 10000000  # Stir Crazy
    train.loc[train["id"] == 1542, "budget"] = 1  # All at Once
    train.loc[train["id"] == 1570, "budget"] = 15800000  # Crocodile Dundee II
    train.loc[train["id"] == 1571, "budget"] = 4000000  # Lady and the Tramp
    train.loc[train["id"] == 1714, "budget"] = 46000000  # The Recruit
    train.loc[train["id"] == 1721, "budget"] = 17500000  # Cocoon
    train.loc[train["id"] == 1865, "revenue"] = 25000000  # Scooby-Doo 2: Monsters Unleashed
    train.loc[train["id"] == 1885, "budget"] = 12  # In the Cut
    train.loc[train["id"] == 2091, "budget"] = 10  # Deadfall
    train.loc[train["id"] == 2268, "budget"] = 17500000  # Madea Goes to Jail budget
    train.loc[train["id"] == 2491, "budget"] = 6  # Never Talk to Strangers
    train.loc[train["id"] == 2602, "budget"] = 31000000  # Mr. Holland's Opus
    train.loc[train["id"] == 2612, "budget"] = 15000000  # Field of Dreams
    train.loc[train["id"] == 2696, "budget"] = 10000000  # Nurse 3-D
    train.loc[train["id"] == 2801, "budget"] = 10000000  # Fracture
    train.loc[train["id"] == 335, "budget"] = 2
    train.loc[train["id"] == 348, "budget"] = 12
    train.loc[train["id"] == 470, "budget"] = 13000000
    train.loc[train["id"] == 513, "budget"] = 1100000
    train.loc[train["id"] == 640, "budget"] = 6
    train.loc[train["id"] == 696, "budget"] = 1
    train.loc[train["id"] == 797, "budget"] = 8000000
    train.loc[train["id"] == 850, "budget"] = 1500000
    train.loc[train["id"] == 1199, "budget"] = 5
    train.loc[train["id"] == 1282, "budget"] = 9  # Death at a Funeral
    train.loc[train["id"] == 1347, "budget"] = 1
    train.loc[train["id"] == 1755, "budget"] = 2
    train.loc[train["id"] == 1801, "budget"] = 5
    train.loc[train["id"] == 1918, "budget"] = 592
    train.loc[train["id"] == 2033, "budget"] = 4
    train.loc[train["id"] == 2118, "budget"] = 344
    train.loc[train["id"] == 2252, "budget"] = 130
    train.loc[train["id"] == 2256, "budget"] = 1
    train.loc[train["id"] == 2696, "budget"] = 10000000

    train = pd.merge(
        train,
        pd.read_csv("comparisons/tmdb_box_office_prediction/TrainAdditionalFeatures.csv"),
        how="left",
        on=["imdb_id"],
    )

    train, test = sklearn.model_selection.train_test_split(train, test_size=0.5)

    train["revenue"] = np.log1p(train["revenue"])
    y = train["revenue"].values

    json_cols = [
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "Keywords",
        "cast",
        "crew",
    ]

    for col in tqdm(json_cols + ["belongs_to_collection"]):
        train[col] = train[col].apply(lambda x: get_dictionary(x))
        test[col] = test[col].apply(lambda x: get_dictionary(x))

    train_dict = get_json_dict(train)
    test_dict = get_json_dict(test)

    # remove cateogry with bias and low frequency
    for col in json_cols:
        remove = []
        train_id = set(list(train_dict[col].keys()))
        test_id = set(list(test_dict[col].keys()))

        remove += list(train_id - test_id) + list(test_id - train_id)
        for i in train_id.union(test_id) - set(remove):
            if train_dict[col][i] < 10 or i == "":
                remove += [i]

        for i in remove:
            if i in train_dict[col]:
                del train_dict[col][i]
            if i in test_dict[col]:
                del test_dict[col][i]

    y_test = test["revenue"].values

    train_copy_for_stats = train.copy(deep=True).reset_index(drop=True)
    all_data = prepare(pd.concat([train, test]).reset_index(drop=True), train_copy_for_stats)
    all_data = all_data.replace([np.inf, -np.inf], np.nan)
    all_data = all_data.fillna(0)

    train = all_data.loc[: train.shape[0] - 1, :]
    test = all_data.loc[train.shape[0] :, :]

    random_seed = 2019
    k = 10
    fold = list(KFold(k, shuffle=True, random_state=random_seed).split(train))
    np.random.seed(random_seed)

    result_dict = dict()
    val_pred = np.zeros(train.shape[0])
    test_pred = np.zeros(test.shape[0])
    final_err = 0
    verbose = False

    for i, (trn, val) in tqdm(enumerate(fold)):
        print(i + 1, "fold.    RMSE")

        trn_x = train.loc[trn, :]
        trn_y = y[trn]
        val_x = train.loc[val, :]
        val_y = y[val]

        fold_val_pred = []
        fold_test_pred = []
        fold_err = []

        # """ xgboost
        start = datetime.now()
        result = xgb_model(trn_x, trn_y, val_x, val_y, test, verbose)
        fold_val_pred.append(result["val"] * 0.2)
        fold_test_pred.append(result["test"] * 0.2)
        fold_err.append(result["error"])
        print(
            "xgb model.",
            "{0:.5f}".format(result["error"]),
            "(" + str(int((datetime.now() - start).seconds / 60)) + "m)",
        )
        # """

        # """ lightgbm
        start = datetime.now()
        result = lgb_model(trn_x, trn_y, val_x, val_y, test, verbose)
        fold_val_pred.append(result["val"] * 0.4)
        fold_test_pred.append(result["test"] * 0.4)
        fold_err.append(result["error"])
        print(
            "lgb model.",
            "{0:.5f}".format(result["error"]),
            "(" + str(int((datetime.now() - start).seconds / 60)) + "m)",
        )
        # """

        # """ catboost model
        start = datetime.now()
        result = cat_model(trn_x, trn_y, val_x, val_y, test, verbose)
        fold_val_pred.append(result["val"] * 0.4)
        fold_test_pred.append(result["test"] * 0.4)
        fold_err.append(result["error"])
        print(
            "cat model.",
            "{0:.5f}".format(result["error"]),
            "(" + str(int((datetime.now() - start).seconds / 60)) + "m)",
        )
        # """

        # mix result of multiple models
        val_pred[val] += np.mean(np.array(fold_val_pred), axis=0)

        test_pred += np.mean(np.array(fold_test_pred), axis=0) / k
        final_err += (sum(fold_err) / len(fold_err)) / k

        print("---------------------------")
        print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))
        print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.mean(np.array(fold_val_pred), axis=0) - val_y) ** 2))))

    rmsle = root_mean_squared_log_error(np.expm1(test_pred * 3), y_test)
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
