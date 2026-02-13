import numpy as np
import pandas as pd
import skrub
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from skrub import selectors as s

import sempipes
from sempipes.optimisers import optimise_colopro
from sempipes.optimisers.evolutionary_search import EvolutionarySearch


def sempipes_pipeline():
    data = skrub.var("data").skb.mark_as_X()
    y = data["SalePrice"]
    data = data.drop(columns=["SalePrice", "Id"])

    y = y.skb.apply_func(np.log).skb.mark_as_y()

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
            parameters={"temperature": 2.0},
        ),
    )

    validation_data = pd.read_csv("experiments/house_prices_advanced_regression_techniques/validation.csv")
    pipeline = sempipes_pipeline()

    outcomes = optimise_colopro(
        pipeline,
        operator_name="house_features",
        search=EvolutionarySearch(population_size=6),
        num_trials=24,
        scoring="neg_root_mean_squared_error",
        cv=10,
        additional_env_variables={"data": validation_data},
    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print(best_outcome.state["generated_code"])
