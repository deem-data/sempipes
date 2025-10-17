# https://www.kaggle.com/code/dway88/feature-eng-feature-importance-random-forest
import ast
import math
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

S = "director_gender"
warnings.filterwarnings("ignore")


def add_datepart(df, field_name):
    """Add date components as separate columns"""
    df[field_name] = pd.to_datetime(df[field_name])
    df[field_name + "_Year"] = df[field_name].dt.year
    df[field_name + "_Month"] = df[field_name].dt.month
    df[field_name + "_Day"] = df[field_name].dt.day
    df[field_name + "_Dayofweek"] = df[field_name].dt.dayofweek
    df[field_name + "_Dayofyear"] = df[field_name].dt.dayofyear
    df[field_name + "_Elapsed"] = (df[field_name] - df[field_name].min()).dt.days
    df.drop(field_name, axis=1, inplace=True)


def train_cats(df):
    """Convert object columns to category type"""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")


def proc_df(df, y_fld):
    """Process dataframe for ML: separate target, handle categoricals, fill NaNs"""
    df = df.copy()
    y = df[y_fld].copy()
    df.drop(y_fld, axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].cat.codes

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)

    return df, y, None


def rf_feat_importance(m, df):
    """Get feature importance from random forest model"""
    return pd.DataFrame({"cols": df.columns, "imp": m.feature_importances_}).sort_values("imp", ascending=False)


scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    np.random.seed(seed)
    data = pd.read_csv("comparisons/tmdb_box_office_prediction/data.csv")
    train_df, test_df = sklearn.model_selection.train_test_split(data, test_size=0.5, random_state=seed)

    df_test = test_df
    df_test.revenue = np.log1p(df_test.revenue)

    df_raw = train_df
    df_raw.revenue = np.log1p(df_raw.revenue)

    # transform release_date to multiple new columns containing information from the date
    add_datepart(df_raw, "release_date")

    mlb = MultiLabelBinarizer()

    def convertStringToList(strVal):
        if type(strVal) is not str:
            return []
        else:
            return ast.literal_eval(strVal)

    def formatDictColumnAndExtractNames(strVal):
        listOfItems = convertStringToList(strVal)
        return list(map(lambda x: x["name"], listOfItems))

    def extractGenres(df):
        df["genres"] = df["genres"].apply(formatDictColumnAndExtractNames)

        return df.join(
            pd.DataFrame(
                mlb.fit_transform(df.pop("genres")),
                columns=list(map(lambda x: "genre_" + x, mlb.classes_)),
                index=df.index,
            )
        )

    df_raw = extractGenres(df_raw)

    def extractCommonProdCompanies(df):
        df["production_companies"] = df["production_companies"].apply(formatDictColumnAndExtractNames)

        companiesCount = df["production_companies"].apply(pd.Series).stack().value_counts()
        companiesToKeep = companiesCount[companiesCount > 30].keys()

        df["production_companies"] = df["production_companies"].apply(
            lambda x: list(filter(lambda i: i in companiesToKeep, x))
        )

        return df.join(
            pd.DataFrame(
                mlb.fit_transform(df.pop("production_companies")),
                columns=list(map(lambda x: "prod_company_" + x, mlb.classes_)),
                index=df.index,
            )
        )

    df_raw = extractCommonProdCompanies(df_raw)

    def extractCommonProdCountries(df):
        df["production_countries"] = df["production_countries"].apply(formatDictColumnAndExtractNames)

        countriesCount = df["production_countries"].apply(pd.Series).stack().value_counts()
        countriesToKeep = countriesCount[countriesCount > 10].keys()

        df["production_countries"] = df["production_countries"].apply(
            lambda x: list(filter(lambda i: i in countriesToKeep, x))
        )
        return df.join(
            pd.DataFrame(
                mlb.fit_transform(df.pop("production_countries")),
                columns=list(map(lambda x: "prod_country_" + x, mlb.classes_)),
                index=df.index,
            )
        )

    df_raw = extractCommonProdCountries(df_raw)

    def extractCommonSpokenLanguages(df):
        df["spoken_languages"] = df["spoken_languages"].apply(formatDictColumnAndExtractNames)

        languageCount = df["spoken_languages"].apply(pd.Series).stack().value_counts()
        languagesToKeep = languageCount[languageCount > 10].keys()

        df["spoken_languages"] = df["spoken_languages"].apply(lambda x: list(filter(lambda i: i in languagesToKeep, x)))

        return df.join(
            pd.DataFrame(
                mlb.fit_transform(df.pop("spoken_languages")),
                columns=list(map(lambda x: "spoken_language_" + x, mlb.classes_)),
                index=df.index,
            )
        )

    df_raw = extractCommonSpokenLanguages(df_raw)

    def extractCommonKeywords(df):
        df["Keywords"] = df["Keywords"].apply(formatDictColumnAndExtractNames)

        keywordCount = df["Keywords"].apply(pd.Series).stack().value_counts()
        keywordsToKeep = keywordCount[keywordCount >= 30].keys()

        df["Keywords"] = df["Keywords"].apply(lambda x: list(filter(lambda i: i in keywordsToKeep, x)))

        return df.join(
            pd.DataFrame(
                mlb.fit_transform(df.pop("Keywords")),
                columns=list(map(lambda x: "keyword_" + x, mlb.classes_)),
                index=df.index,
            )
        )

    df_raw = extractCommonKeywords(df_raw)

    def addCastLengthColumn(df):
        castNames = df["cast"].apply(formatDictColumnAndExtractNames)
        df["cast_len"] = castNames.apply(lambda x: len(x))
        return df

    df_raw = addCastLengthColumn(df_raw)
    df_raw.drop(["cast"], axis=1, inplace=True)

    def formatDictColumnAndExtractJobName(strVal, job):
        listOfItems = convertStringToList(strVal)

        jobItem = (list(filter(lambda lst: lst["job"] == job, listOfItems)) or [None])[0]
        if type(jobItem) is dict:
            return jobItem["name"]
        else:
            return None

    def addCrewJobsColumns(df):
        df["director"] = df["crew"].apply(formatDictColumnAndExtractJobName, args=("Director",))
        df["screenplay"] = df["crew"].apply(formatDictColumnAndExtractJobName, args=("Screenplay",))
        df["director_of_photography"] = df["crew"].apply(
            formatDictColumnAndExtractJobName, args=("Director of Photography",)
        )
        df["original_music_composer"] = df["crew"].apply(
            formatDictColumnAndExtractJobName, args=("Original Music Composer",)
        )
        df["art_director"] = df["crew"].apply(formatDictColumnAndExtractJobName, args=("Art Direction",))

        return df

    df_raw = addCrewJobsColumns(df_raw)

    def formatDictColumnAndExtractDirectorGender(strVal):
        listOfItems = convertStringToList(strVal)

        directorItem = (list(filter(lambda lst: lst["job"] == "Director", listOfItems)) or [None])[0]
        if type(directorItem) is dict:
            return directorItem["gender"]
        else:
            return None

    def addDirectorGenderColumn(df):
        df[("%s" % S)] = df["crew"].apply(formatDictColumnAndExtractDirectorGender)
        return df

    df_raw = addDirectorGenderColumn(df_raw)

    def addCrewLenghtColumn(df):
        df["crew"] = df["crew"].apply(formatDictColumnAndExtractNames)
        df["crew_len"] = df["crew"].apply(lambda x: len(x))
        return df

    df_raw = addCrewLenghtColumn(df_raw)
    # drop crew column
    df_raw.drop(["crew"], axis=1, inplace=True)

    df_raw["has_homepage"] = df_raw["homepage"].apply(lambda x: isinstance(x, str))

    df_raw["belongs_to_collection"] = df_raw["belongs_to_collection"].apply(lambda x: isinstance(x, str))

    def extractTaglineInfo(df):
        df["has_tagline"] = df["tagline"].apply(lambda x: isinstance(x, str))
        df["tagline_len"] = df["tagline"].apply(lambda x: len(x) if isinstance(x, str) else 0)
        return df

    df_raw = extractTaglineInfo(df_raw)

    def extractOverviewInfo(df):
        df["has_overview"] = df["overview"].apply(lambda x: isinstance(x, str))
        df["overview_len"] = df["overview"].apply(lambda x: len(x) if isinstance(x, str) else 0)
        return df

    df_raw = extractOverviewInfo(df_raw)

    # we noticed quite a lot of movies with budget 0...
    df_raw["has_budget"] = df_raw["budget"].apply(lambda x: x > 0)

    toRemove = ["imdb_id", "id", "poster_path", "overview", "homepage", "tagline", "original_title", "status"]
    df_raw.drop(toRemove, axis=1, inplace=True)

    train_cats(df_raw)

    df_trn, y_trn, nas = proc_df(df_raw, "revenue")
    m = RandomForestRegressor(n_jobs=-1)
    m.fit(df_trn, y_trn)
    m.score(df_trn, y_trn)

    def split_vals(a, n):
        return a[:n], a[n:]

    n_valid = 600  # 20%
    n_trn = len(df_trn) - n_valid
    X_train, X_valid = split_vals(df_trn, n_trn)
    y_train, y_valid = split_vals(y_trn, n_trn)
    raw_train, raw_valid = split_vals(df_raw, n_trn)

    def rmse(x, y):
        return math.sqrt(((x - y) ** 2).mean())

    # rf with hyper parameters
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)

    fi = rf_feat_importance(m, df_trn)
    to_keep = fi[fi.imp > 0.002].cols
    df_keep = df_trn[to_keep].copy()
    # to_drop = ['release_Month', 'release_Elapsed']\

    X_train, X_valid = split_vals(df_keep, n_trn)
    y_train, y_valid = split_vals(y_trn, n_trn)

    m = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)

    add_datepart(df_test, "release_date")

    df_test = extractGenres(df_test)
    df_test = extractCommonProdCompanies(df_test)
    df_test = extractCommonProdCountries(df_test)
    df_test = extractCommonSpokenLanguages(df_test)
    df_test = extractCommonKeywords(df_test)
    df_test = addCastLengthColumn(df_test)
    df_test.drop(["cast"], axis=1, inplace=True)
    df_test = addCrewJobsColumns(df_test)
    df_test = addDirectorGenderColumn(df_test)
    df_test = addCrewLenghtColumn(df_test)
    df_test.drop(["crew"], axis=1, inplace=True)
    df_test["has_homepage"] = df_test["homepage"].apply(lambda x: isinstance(x, str))
    df_test["belongs_to_collection"] = df_test["belongs_to_collection"].apply(lambda x: isinstance(x, str))
    df_test = extractTaglineInfo(df_test)
    df_test = extractOverviewInfo(df_test)
    df_test["has_budget"] = df_test["budget"].apply(lambda x: x > 0)
    df_test.drop(toRemove, axis=1, inplace=True)

    train_cats(df_test)
    df_test, _, _ = proc_df(df_test, "revenue")
    df_test_keep = df_test[to_keep].copy()
    predictions = m.predict(df_test_keep)

    from sklearn.metrics import mean_squared_error

    rmsle = np.sqrt(mean_squared_error(test_df["revenue"], predictions))
    print(f"RMSLE on split {split_index}: {rmsle}")
    scores.append(rmsle)

print("\nMean final score: ", np.mean(scores), np.std(scores))
