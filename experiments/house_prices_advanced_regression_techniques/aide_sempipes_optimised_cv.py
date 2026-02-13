import math

import numpy as np
import pandas as pd
import skrub
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skrub import selectors as s

import sempipes

best_state = {
    "generated_code": [
        "",
        "# Total_Living_Area (Sum of above ground living area and total basement area)\n# Input samples: 'GrLivArea': [1144, 2520, 1520], 'TotalBsmtSF': [864, 1338, 793]\ndf['Total_Living_Area'] = df['GrLivArea'] + df['TotalBsmtSF']\n\n# House_Age (Age of the house at the time of sale)\n# Input samples: 'YrSold': [2009, 2006, 2009], 'YearBuilt': [1961, 1993, 1932]\ndf['House_Age'] = df['YrSold'] - df['YearBuilt']\n\n# Years_Since_Last_Update (Years since the last remodel or new construction at time of sale)\n# Input samples: 'YrSold': [2009, 2006, 2009], 'YearRemodAdd': [1983, 1993, 2000]\ndf['Years_Since_Last_Update'] = df['YrSold'] - df['YearRemodAdd']\n\n# Total_Weighted_Bathrooms (Combined count of all full and half bathrooms, weighting half baths as 0.5)\n# Input samples: 'BsmtFullBath': [1, 1, 0], 'BsmtHalfBath': [0, 0, 0], 'FullBath': [1, 2, 1], 'HalfBath': [0, 1, 0]\ndf['Total_Weighted_Bathrooms'] = df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'] + df['FullBath'] + 0.5 * df['HalfBath']\n\n# Overall_Quality_Index (Combined metric for overall quality and condition)\n# Input samples: 'OverallQual': [5, 8, 5], 'OverallCond': [6, 7, 5]\ndf['Overall_Quality_Index'] = df['OverallQual'] * df['OverallCond']\n\n# Has_Garage (Binary indicator if the house has a garage)\n# Input samples: 'GarageCars': [1, 3, 1]\ndf['Has_Garage'] = (df['GarageCars'] > 0).astype(int)\n\n# Has_Pool (Binary indicator if the house has a pool)\n# Input samples: 'PoolArea': [0, 0, 0]\ndf['Has_Pool'] = (df['PoolArea'] > 0).astype(int)\n\n# Total_Outdoor_SF (Total area of all outdoor porches and decks)\n# Input samples: 'WoodDeckSF': [165, 209, 0], 'OpenPorchSF': [0, 55, 0], 'EnclosedPorch': [0, 0, 56], '3SsnPorch': [0, 0, 0], 'ScreenPorch': [0, 0, 0]\ndf['Total_Outdoor_SF'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']",
        "import numpy as np\n\n# Re-establish quality mapping to be sure it's available\nquality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1} \n\n# (LotFrontage_Imputed - Lot frontage, with NaNs imputed by neighborhood median)\n# Input samples: 'LotFrontage': [80.0, nan, 55.0], 'Neighborhood': ['Sawyer', 'NoRidge', 'SWISU']\ndf['LotFrontage_Imputed'] = df['LotFrontage'].fillna(df.groupby('Neighborhood')['LotFrontage'].transform('median'))\n# Fallback for neighborhoods not present in training data for median calculation, use overall median\ndf['LotFrontage_Imputed'] = df['LotFrontage_Imputed'].fillna(df['LotFrontage'].median())\n\n# (Log_LotArea - Log transformation of LotArea for handling skewed distribution)\n# Input samples: 'LotArea': [10000, 14541, 4500]\ndf['Log_LotArea'] = np.log1p(df['LotArea'])\n\n# (Log_GrLivArea - Log transformation of Ground Living Area for handling skewed distribution)\n# Input samples: 'GrLivArea': [1144, 2520, 1520]\ndf['Log_GrLivArea'] = np.log1p(df['GrLivArea'])\n\n# (Total_Floor_SF - Total square footage of all floors including basement, 1st and 2nd)\n# Input samples: 'TotalBsmtSF': [864, 1338, 793], '1stFlrSF': [1144, 1352, 848], '2ndFlrSF': [0, 1168, 672]\ndf['Total_Floor_SF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']\n\n# (Area_Per_Room - Total square footage per room above grade)\n# Input samples: 'Total_Living_Area': [2008, 3858, 2313], 'TotRmsAbvGrd': [6, 10, 6]\n# Total_Living_Area should exist from the first block\ndf['Area_Per_Room'] = df['Total_Living_Area'] / df['TotRmsAbvGrd']\ndf['Area_Per_Room'] = df['Area_Per_Room'].replace([np.inf, -np.inf], 0).fillna(0) # Handle division by zero\n\n# (Overall_SF_Quality - Interaction between total square footage and overall quality)\n# Input samples: 'Total_Living_Area': [2008, 3858, 2313], 'OverallQual': [5, 8, 5]\ndf['Overall_SF_Quality'] = df['Total_Living_Area'] * df['OverallQual']\n\n# (Garage_Quality_Area - Interaction between Garage Area and mapped Garage Quality)\n# Input samples: 'GarageArea': [264, 796, 281], 'GarageQual': ['TA', 'TA', 'TA']\ndf['GarageQual_Score'] = df['GarageQual'].map(quality_map).fillna(0) # Redefine here to ensure it exists\ndf['Garage_Quality_Area'] = df['GarageArea'] * df['GarageQual_Score']\n\n# (Total_Porch_SF - Sum of all different types of porch square footage)\n# Input samples: 'OpenPorchSF': [0, 55, 0], 'EnclosedPorch': [0, 0, 56], '3SsnPorch': [0, 0, 0], 'ScreenPorch': [0, 0, 0]\ndf['Total_Porch_SF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']\n\n# (Is_Remodeled - Binary indicator if the house was remodeled after its initial construction)\n# Input samples: 'YearBuilt': [1961, 1993, 1932], 'YearRemodAdd': [1983, 1993, 2000]\ndf['Is_Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)\n\n# (Has_Basement - Binary indicator if the house has a basement)\n# Input samples: 'TotalBsmtSF': [864, 1338, 793]\ndf['Has_Basement'] = (df['TotalBsmtSF'] > 0).astype(int)",
        "import numpy as np\n\n# Re-define standard quality/condition mapping for ordinal features\n# Mapping 'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1. NaN/None usually implies absence of the feature or worst condition, mapped to 0.\nquality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, None: 0, np.nan: 0}\n\n# Mapping for Functional (Typ: Typical to Sal: Severely Damaged)\nfunctional_map = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0, None: 0, np.nan: 0}\n\n# (ExterQual_Score - Exterior material quality, mapped to numerical scores)\n# Input samples: 'ExterQual': ['TA', 'Gd', 'TA']\ndf['ExterQual_Score'] = df['ExterQual'].map(quality_map).fillna(0)\n\n# (BsmtQual_Score - Basement height, mapped to numerical scores. NaN means no basement.)\n# Input samples: 'BsmtQual': ['TA', 'Gd', 'TA', 'Gd', 'TA']\ndf['BsmtQual_Score'] = df['BsmtQual'].map(quality_map).fillna(0)\n\n# (KitchenQual_Score - Kitchen quality, mapped to numerical scores)\n# Input samples: 'KitchenQual': ['TA', 'Gd', 'TA']\ndf['KitchenQual_Score'] = df['KitchenQual'].map(quality_map).fillna(0)\n\n# (HeatingQC_Score - Heating quality and condition, mapped to numerical scores)\n# Input samples: 'HeatingQC': ['Ex', 'Ex', 'Ex']\ndf['HeatingQC_Score'] = df['HeatingQC'].map(quality_map).fillna(0)\n\n# (FireplaceQu_Score - Fireplace quality, mapped to numerical scores. NaN means no fireplace.)\n# Input samples: 'FireplaceQu': ['TA', 'TA', nan]\ndf['FireplaceQu_Score'] = df['FireplaceQu'].map(quality_map).fillna(0)\n\n# (GarageQual_Score_explicit - Garage quality, explicitly re-mapped for clarity. NaN means no garage.)\n# Input samples: 'GarageQual': ['TA', 'TA', 'TA']\ndf['GarageQual_Score_explicit'] = df['GarageQual'].map(quality_map).fillna(0)\n\n# (Functional_Score - Home functionality, mapped to numerical scores)\n# Input samples: 'Functional': ['Typ', 'Typ', 'Typ']\ndf['Functional_Score'] = df['Functional'].map(functional_map).fillna(0)\n\n# (Age_SqFt_Interaction - Interaction feature combining house age and above ground living area)\n# Input samples: 'House_Age': [48, 13, 77], 'GrLivArea': [1144, 2520, 1520]\ndf['Age_SqFt_Interaction'] = df['House_Age'] * df['GrLivArea']\n\n# (Log_LotFrontage - Log transformation of Lot frontage for skewed distribution)\n# Input samples: 'LotFrontage_Imputed': [80.0, 75.0, 55.0]\ndf['Log_LotFrontage'] = np.log1p(df['LotFrontage_Imputed'])\n\n# (YearBuilt_OverallQual_Interaction - Interaction of build year with overall quality, highlighting quality of older vs newer homes)\n# Input samples: 'YearBuilt': [1961, 1993, 1932], 'OverallQual': [5, 8, 5]\ndf['YearBuilt_OverallQual_Interaction'] = df['YearBuilt'] * df['OverallQual']",
    ]
}


def sempipes_pipeline():
    data = skrub.var("data").skb.mark_as_X()
    y = data["SalePrice"].skb.mark_as_y()
    data = data.drop(columns=["SalePrice", "Id"])

    y = y.skb.apply_func(np.log)

    def interaction_terms(df):
        key_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"]
        for i in range(len(key_features)):
            for j in range(i + 1, len(key_features)):
                name = key_features[i] + "_X_" + key_features[j]
                df[name] = df[key_features[i]] * df[key_features[j]]
        return df

    data = data.skb.apply_func(interaction_terms)

    data = data.sem_gen_features(
        nl_prompt="""
        Create additional features that could help predict the sale price of a house.
        Consider aspects like location, size, condition, and any other relevant information.
        You way want to combine several existing features to create new ones.
        """,
        name="house_features",
        how_many=10,
    )

    data = data.skb.apply(SimpleImputer(strategy="median"), cols=s.numeric())
    data = data.skb.apply(SimpleImputer(strategy="most_frequent"), cols=s.categorical())

    data = data.skb.apply(skrub.TableVectorizer(numeric=StandardScaler()))
    model = Lasso(alpha=0.001, random_state=42)
    predictions = data.skb.apply(model, y=y)
    return predictions


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 0.0},
        ),
    )

    scores = []
    seed = 42
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        train = all_data.iloc[train_idx]
        test = all_data.iloc[test_idx]
        np.random.seed(seed)
        # Load the data
        # all_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/data.csv")
        # train, test = train_test_split(all_data, test_size=0.5, random_state=seed)

        y_true = np.log(test["SalePrice"])

        predictions = sempipes_pipeline()
        learner = predictions.skb.make_learner(fitted=False, keep_subsampling=False)

        env_train = predictions.skb.get_data()
        env_train["data"] = train
        env_train["sempipes_prefitted_state__house_features"] = best_state
        env_test = predictions.skb.get_data()
        env_test["data"] = test
        env_test["sempipes_prefitted_state__house_features"] = best_state

        learner.fit(env_train)
        y_pred = learner.predict(env_test)

        score = math.sqrt(mean_squared_error(y_true, y_pred))
        print(f"RMSLE on split {fold_index}: {score}")
        scores.append(score)

    print("\nMean final score: ", np.mean(scores), np.std(scores))
