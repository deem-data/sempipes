import logging
import warnings

import numpy as np
import pandas as pd
import skrub
from xgboost import XGBClassifier

import sempipes
from experiments.sivep import DATA_DESCRIPTION
from sempipes.optimisers import optimise_colopro
from sempipes.optimisers.montecarlo_tree_search import MonteCarloTreeSearch

warnings.filterwarnings("ignore")

logging.getLogger("sdv").setLevel(logging.ERROR)


def _pipeline(raw_data, seed):
    data = skrub.var("data", raw_data)

    patients = data.drop(columns="due_to_covid", errors="ignore").skb.set_description(DATA_DESCRIPTION).skb.mark_as_X()

    labels = (
        data["due_to_covid"]
        .skb.set_description("Indicator whether the severe acute respiratory infections originated from COVID-19.")
        .skb.mark_as_y()
    )

    data_to_augment = patients.assign(due_to_covid=labels)

    augmented_data = data_to_augment.sem_augment(
        nl_prompt="""
        Augment the dataset with additional records similar to the existing records of people from the indegenous minority in Brazil, for whom the prediction model may not work as well as for the majority.
        The records of the indegenous minority have the `cs_raca` column set to 5. The additional data should improve the prediction quality for them, so make sure that it follows the same distribution as the original data.""",
        name="augment_data",
        number_of_rows_to_generate=600,
        generate_via_code=True,
    )

    def only_indegenous_for_evaluation(df, eval_mode=skrub.eval_mode()):
        if eval_mode == "predict" or eval_mode == "score":
            filtered_df = df[df.cs_raca == 5]
            return filtered_df
        else:
            return df

    augmented_data_maybe_filtered = augmented_data.skb.apply_func(only_indegenous_for_evaluation)

    X = augmented_data_maybe_filtered.drop(columns="due_to_covid", errors="ignore")
    y = augmented_data_maybe_filtered["due_to_covid"]

    X_encoded = X.skb.apply(skrub.TableVectorizer())

    predictions = X_encoded.skb.apply(XGBClassifier(eval_metric="logloss", random_state=seed), y=y)

    return predictions


if __name__ == "__main__":
    df = pd.read_csv("experiments/sivep/validation.csv")
    df = df.sample(frac=0.5, random_state=42)

    # Remove records with unreported race
    df = df[df.cs_raca != 9]
    # Remove influenza cases
    df = df[~df.classi_fin.isin([1])]
    # Target label: SRAG due to covid
    df["due_to_covid"] = df.classi_fin == 5

    df = df.drop(columns=["classi_fin", "evolucao", "vacina_cov", "cs_sexo", "dt_evoluca", "dt_interna"])

    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
    )

    pipeline = _pipeline(df, 42)

    np.random.seed(42)

    outcomes = optimise_colopro(
        pipeline,
        operator_name="augment_data",
        search=MonteCarloTreeSearch(c=0.5),
        num_trials=24,
        scoring="roc_auc",
        cv=5,
        run_name="sivep",
    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print(best_outcome.state["generated_code"])
