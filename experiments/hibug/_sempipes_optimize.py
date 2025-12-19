import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator

import sempipes
from sempipes.optimisers import EvolutionarySearch, optimise_colopro

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 2.0},
    ),
)


class NoPredictor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X


def annotation_pipeline():
    samples_to_annotate = skrub.var("samples_to_annotate").skb.mark_as_X()
    dummy_y = skrub.var("dummy_y").skb.mark_as_y()

    annotated = samples_to_annotate.sem_extract_features(
        nl_prompt="""
        Annotate a set of celebrity images with the specified attributes. The attributes will be used to debug a model predicting whether a celebrity is wearing lipstick in an image. Make sure that your attributes are correlated with potential failures of this prediction task.

        IMPORTANT: Each attribute value should consist of a single word or phrase only from the list of potential answers!. 
        """,
        input_columns=["image"],
        name="extract_features",
        output_columns={
            "beard": "Does the person have a beard?",
            "makeup": "Does the person wear makeup?",
            "gender": "Is the person in the photo a male or a female?",
            "hair_color": "Which of the following hair colors does the person in the photo have: blonde, brown, black, gray, white or red?",
            "skin_color": "Does the person in the photo have white, brown or black skin?",
            "emotion": "Which of the following emotions is the person in the photo showing: sad, serious, calm, happy, surprised, neutral, angry, excited, pensive?",
            "age": "Which of the following age ranges is the person in the photo in: young, middle-aged, old?",
        },
        generate_via_code=True,
        print_code_to_console=True,
    )

    return annotated.skb.apply(NoPredictor(), y=dummy_y)


df = pd.read_csv("experiments/hibug/hibug_attributes.csv")
df = df.iloc[:2000]
df = df[["idx", "image", "label", "prediction"]]


def score_annotations(estimator, X, y=None, **kwargs):
    # print(X.head())
    to_score = estimator.predict(X)
    # print(to_score.head())
    if len(to_score.columns) <= 4:
        return 0.0001

    from experiments.hibug.scoring import score_sempipes

    accuracy, _ = score_sempipes(to_score)
    return accuracy


outcomes = optimise_colopro(
    dag_sink=annotation_pipeline(),
    operator_name="extract_features",
    num_trials=24,
    scoring=score_annotations,
    search=EvolutionarySearch(population_size=6),
    cv=2,
    additional_env_variables={
        "samples_to_annotate": df,
        "dummy_y": np.zeros(len(df)),
    },
)

best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
print(best_outcome.state["generated_code"])
