import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from skrub import SelectCols
from skrub import selectors as s
from tqdm import tqdm

import sempipes  # pylint: disable=unused-import

warnings.filterwarnings("ignore")


def xgb_model(trn_x, trn_y, val_x, val_y, random_seed):
    params = {
        "objective": "reg:squarederror",
        "eta": 0.01,
        "max_depth": 6,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "eval_metric": "rmse",
        "seed": random_seed,
        "verbosity": 0,
    }

    evals_result = {}
    model = xgb.train(
        params,
        xgb.DMatrix(trn_x, trn_y),
        100000,
        evals=[(xgb.DMatrix(trn_x, trn_y), "train"), (xgb.DMatrix(val_x, val_y), "valid")],
        verbose_eval=False,
        early_stopping_rounds=500,
        evals_result=evals_result,
    )

    def predictor(test):
        return model.predict(xgb.DMatrix(test), iteration_range=(0, model.best_iteration))

    return predictor


def lgb_model(trn_x, trn_y, val_x, val_y, random_seed):
    params = {
        "objective": "regression",
        "num_leaves": 30,
        "min_data_in_leaf": 20,
        "max_depth": 9,
        "learning_rate": 0.004,
        "feature_fraction": 0.9,
        "bagging_freq": 1,
        "bagging_fraction": 0.9,
        "lambda_l1": 0.2,
        "bagging_seed": random_seed,
        "metric": "rmse",
        "random_state": random_seed,
        "verbosity": -1,
    }

    record = {}
    model = lgb.train(
        params,
        lgb.Dataset(trn_x, trn_y),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(val_x, val_y)],
        callbacks=[lgb.record_evaluation(record), lgb.log_evaluation(False)],
    )

    def predictor(test):
        return model.predict(test, num_iteration=model.best_iteration)

    return predictor


def cat_model(trn_x, trn_y, val_x, val_y, random_seed):
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

    def predictor(test):
        return model.predict(test)

    return predictor


class CrazyEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.num_folds = 10
        self.xgb_predictors_ = []
        self.lgb_predictors_ = []
        self.cat_predictors_ = []

    def fit(self, train, y):
        random_seed = 2019
        np.random.seed(random_seed)
        fold = list(KFold(self.num_folds, shuffle=True, random_state=random_seed).split(train))

        # Reset index to ensure integer-based indexing works
        train = train.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for trn, val in tqdm(fold):
            trn_x = train.iloc[trn, :]
            trn_y = y.iloc[trn]
            val_x = train.iloc[val, :]
            val_y = y.iloc[val]

            self.xgb_predictors_.append(xgb_model(trn_x, trn_y, val_x, val_y, random_seed))
            self.lgb_predictors_.append(lgb_model(trn_x, trn_y, val_x, val_y, random_seed))
            self.cat_predictors_.append(cat_model(trn_x, trn_y, val_x, val_y, random_seed))

        return self

    def predict(self, test):
        test_pred = np.zeros(test.shape[0])
        for k in range(self.num_folds):
            fold_test_pred = []
            fold_test_pred.append(self.xgb_predictors_[k](test) * 0.2)
            fold_test_pred.append(self.lgb_predictors_[k](test) * 0.4)
            fold_test_pred.append(self.cat_predictors_[k](test) * 0.4)
            test_pred += np.mean(np.array(fold_test_pred), axis=0) / 10
        return test_pred


def sempipes_pipeline3(data_file: str):
    data = pd.read_csv(data_file)
    revenue_df = data.revenue
    data = data.drop(columns=["revenue"])

    data = pd.merge(
        data,
        pd.read_csv("experiments/tmdb_box_office_prediction/TrainAdditionalFeatures.csv"),
        how="left",
        on=["imdb_id"],
    )

    def corrections(df):
        budget_corrections = {
            90: 30000000,  # Sommersby
            118: 60000000,  # Wild Hogs
            149: 18000000,  # Beethoven
            464: 20000000,  # Parenthood
            470: 13000000,  # The Karate Kid, Part II
            513: 1100000,  # From Prada to Nada (updated value)
            797: 8000000,  # Welcome to Dongmakgol
            819: 90000000,  # Alvin and the Chipmunks: The Road Chip
            850: 1500000,  # Modern Times (updated value)
            1007: 2,  # Zyzzyx Road
            1112: 7500000,  # An Officer and a Gentleman
            1131: 4300000,  # Smokey and the Bandit
            1359: 10000000,  # Stir Crazy
            1542: 1,  # All at Once
            1570: 15800000,  # Crocodile Dundee II
            1571: 4000000,  # Lady and the Tramp
            1714: 46000000,  # The Recruit
            1721: 17500000,  # Cocoon
            1885: 12,  # In the Cut
            2091: 10,  # Deadfall
            2268: 17500000,  # Madea Goes to Jail
            2491: 6,  # Never Talk to Strangers
            2602: 31000000,  # Mr. Holland's Opus
            2612: 15000000,  # Field of Dreams
            2696: 10000000,  # Nurse 3-D
            2801: 10000000,  # Fracture
            335: 2,
            348: 12,
            640: 6,
            696: 1,
            1199: 5,
            1282: 9,  # Death at a Funeral
            1347: 1,
            1755: 2,
            1801: 5,
            1918: 592,
            2033: 4,
            2118: 344,
            2252: 130,
            2256: 1,
        }
        # Apply corrections
        for movie_id, budget in budget_corrections.items():
            df.loc[df["id"] == movie_id, "budget"] = budget
        return df

    movie_stats = skrub.var("movie_stats", data).skb.mark_as_X().skb.subsample(n=100)
    movie_stats = movie_stats.skb.set_description("""
        In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
    """)

    movie_stats = movie_stats.skb.apply_func(corrections)

    revenue = skrub.var("revenue", revenue_df).skb.mark_as_y().skb.subsample(n=100)
    revenue = revenue.skb.set_description("the international box office revenue for a movie")
    y_log = revenue.skb.apply_func(np.log1p)

    movie_stats = movie_stats.with_sem_features(
        nl_prompt="""
        Create additional features that could help predict the box office revenue of a movie. Here are detailed instructions:

        Compute the year, month, day, day of week, day of year of the movie release date, make sure that the year is four digits.

        For movies without a rating, impute the mean rating from movies with the same release year and original language.
        For movies without a totalVotes, impute the mean totalVotes from movies with the same release year and original language.

        Compute an inflation adjust budget feature and a log-normalized budget feature.
        
        Compute the number of crew members with a particular gender, the collection name, number of keywords, number of cast members.
        
        Compute the mean popularity of movies in the same year, the ratio of budget to runtime, the ratio of budget to popularity, the ration of totalVotes to popularity, the ratio of rating to popularity.

        Extract the most common spoken languages, production countries, production companies, and genres as separate features.  

        Compute whether the movie has a homepage, whether it belongs to a collection, whether the tagline is missing, whether the original language is English, whether the original title is different from the title, whether the movie is released, as well as the number of letters and words in the original title.

        Moreover, compute the number of words in the title, overview, and tagline, the number of production companies, the number of production countries, the number of cast members, and the number of crew members. Also compute the mean runtime for the movies year, the mean popularity for the movies year, the mean budget for the movies year, the mean totalVotes for the movies year, the mean totalVotes for the movies rating, and the median budget for the movies year.

        REPLACE ALL CATEGRORICAL FEATURES WITH ORDINAL ENCODING, MAKE SURE THAT THE ORDER IS MEANINGFUL.
        """,
        name="additional_movie_features",
        how_many=50,
    )

    to_remove = [
        "imdb_id",
        "id",
        "poster_path",
        "overview",
        "homepage",
        "tagline",
        "original_title",
        "status",
        "cast",
        "release_date",
        "Keywords",
        "crew",
        "belongs_to_collection",
        "spoken_languages",
        "production_countries",
        "production_companies",
        "genres",
    ]
    movie_stats = movie_stats.skb.apply(SelectCols(s.inv(to_remove)))
    movie_stats = movie_stats.replace([np.inf, -np.inf], np.nan)
    movie_stats = movie_stats.fillna(0)

    X = movie_stats.skb.apply(SelectCols(s.numeric() | s.boolean()))

    ensemble = CrazyEnsemble()
    predictions = X.skb.apply(ensemble, y=y_log)

    def exp_if_transform(outputs, mode=skrub.eval_mode()):
        if mode in {"transform", "predict"}:
            return np.expm1(outputs * 3)
        return outputs

    return predictions.skb.apply_func(exp_if_transform)
