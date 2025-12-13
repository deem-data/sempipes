import json
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import sempipes
from experiments.sivep import DATA_DESCRIPTION

warnings.filterwarnings("ignore")


with open("experiments/sivep/_sempipes_state.json", "r", encoding="utf-8") as f:
    best_state = json.load(f)


def _pipeline(raw_data, seed):
    data = skrub.var("data", raw_data).skb.set_description(DATA_DESCRIPTION)

    data_augmented = data.sem_augment(
        nl_prompt="""
        Augment the dataset with additional records similar to the existing records of people from the indegenous minority in Brazil, for whom the prediction model may not work as well as for the majority. The records of the indegenous minority have the `cs_raca` column set to 5 The additional data should improve the prediction quality for them, so make sure that it follows the same distribution as the original data.""",
        name="augment_data",
        number_of_rows_to_generate=600,
        generate_via_code=True,
    )

    X = data_augmented.drop(columns="due_to_covid", errors="ignore").skb.mark_as_X()
    y = (
        data_augmented["due_to_covid"]
        .skb.mark_as_y()
        .skb.set_description("Indicator whether the severe acute respiratory infections originated from COVID-19.")
    )

    X_encoded = X.skb.apply(skrub.TableVectorizer())

    return X_encoded.skb.apply(XGBClassifier(eval_metric="logloss", random_state=seed), y=y)


sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 0.0},
    ),
)

all_data = pd.read_csv("experiments/sivep/data.csv")

scores = []
for seed in [42, 1337, 2025, 7321, 98765]:
    np.random.seed(seed)
    df = all_data.sample(frac=0.1, random_state=seed)

    # Remove records with unreported race
    df = df[df.cs_raca != 9]
    # Remove influenza cases
    df = df[~df.classi_fin.isin([1])]
    # Target label: SRAG due to covid
    df["due_to_covid"] = df.classi_fin == 5

    data = df.drop(columns=["classi_fin", "evolucao", "vacina_cov", "cs_sexo", "dt_evoluca", "dt_interna"])
    train, test = train_test_split(data, test_size=0.5, random_state=seed)

    pipeline = _pipeline(data, seed)

    learner = pipeline.skb.make_learner()

    from sempipes.inspection.pipeline_summary import summarise_pipeline

    pipeline_summary = summarise_pipeline(pipeline, _pipeline)

    env_train = {
        "data": train,
        "sempipes_pipeline_summary__augment_data": pipeline_summary,
        "sempipes_prefitted_state__augment_data": best_state,
        "sempipes_memory__augment_data": [],
        "sempipes_inspirations__augment_data": [],
    }

    learner.fit(env_train)

    majority_groups = {1, 2, 3, 4}
    test_minority = test[~test.cs_raca.isin(majority_groups)]
    test_minority_labels = test_minority.due_to_covid

    env_test = {
        "data": test_minority,
        "sempipes_pipeline_summary__augment_data": pipeline_summary,
        "sempipes_prefitted_state__augment_data": None,
        "sempipes_memory__augment_data": [],
        "sempipes_inspirations__augment_data": [],
    }

    predictions = learner.predict_proba(env_test)
    augmented_minority_score = roc_auc_score(test_minority_labels, predictions[:, 1])

    print(f"ROC AUC score for minority group on seed {seed}: {augmented_minority_score}")
    scores.append(augmented_minority_score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
