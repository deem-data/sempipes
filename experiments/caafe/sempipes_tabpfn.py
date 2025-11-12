import json
import random
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import sempipes
from experiments.caafe import scoring
from sempipes.optimisers import TreeSearch
from sempipes.optimisers.colopro import optimise_colopro

warnings.filterwarnings("ignore")


sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 2.0},
    ),
)


def with_sempipes(X, y, description, target_name, seed):
    X_var = skrub.var("X", X)
    X_var = sempipes.as_X(X_var, description)
    y_var = skrub.var("y", y)
    y_var = sempipes.as_y(y_var, target_name)

    X_with_additional_features = X_var.sem_gen_features(
        nl_prompt="",
        name="additional_features",
        how_many=1,
    )
    X_with_additional_features = X_with_additional_features.replace([np.inf, -np.inf, np.nan], 0)
    encoded = X_with_additional_features.skb.apply(TableVectorizer())

    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    return encoded.skb.apply(clf, y=y_var)


dataset_name = "airlines"

baselines_scores = []
sempipes_scores = []

for seed in range(0, 5):
    print(f"######################## Starting round {seed}")
    np.random.seed(seed)
    random.seed(seed)

    data = pd.read_csv(f"experiments/caafe/data/{dataset_name}/data.csv")
    target_column = data.columns[-1]

    train, test = train_test_split(data, test_size=0.25, random_state=seed)

    y_train = train[target_column]
    X_train = train.drop(columns=[target_column])
    X_train = X_train.fillna(0)

    y_test = test[target_column]
    X_test = test.drop(columns=[target_column])
    X_test = X_test.fillna(0)

    baseline_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)

    X_train_baseline = X_train.copy(deep=True)
    X_test_baseline = X_test.copy(deep=True)

    baseline_clf.fit(X_train_baseline, y_train)

    baseline_y_pred = baseline_clf.predict_proba(X_test_baseline)
    baseline_score = scoring(y_test, baseline_y_pred)
    baselines_scores.append(baseline_score)
    print("Baseline ROC AUC:", baseline_score)

    with open(f"experiments/caafe/data/{dataset_name}/metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    pipeline = with_sempipes(X_train, y_train, metadata["description"], target_column, seed)

    outcomes = optimise_colopro(
        pipeline,
        operator_name="additional_features",
        num_trials=24,
        scoring="roc_auc_ovo",
        search=TreeSearch(),
        cv=10,
        pipeline_definition=with_sempipes,
        run_name=dataset_name,
    )
    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))  # type: ignore[operator]
    state = best_outcome.state

    print("STATE")
    print(state)
    # Write to a JSON file
    with open(f"{dataset_name}_{seed}_code.json", "w") as f:
        json.dump(state, f, indent=4)

    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    env_train = {
        "_skrub_X": X_train,
        "_skrub_y": y_train,
        "sempipes_memory__additional_features": None,
        "sempipes_pipeline_summary__additional_features": None,
        "sempipes_prefitted_state__additional_features": state,
    }
    learner.fit(env_train)

    env_test = {
        "_skrub_X": X_test,
        "_skrub_y": None,
        "sempipes_memory__additional_features": None,
        "sempipes_pipeline_summary__additional_features": None,
        "sempipes_prefitted_state__additional_features": state,
    }
    y_pred = learner.predict_proba(env_test)

    score = scoring(y_test, y_pred)
    sempipes_scores.append(score)
    print("Sempipes ROC AUC:", score)

baseline_mean = np.mean(baselines_scores)
baseline_std = np.std(baselines_scores)
sempipes_mean = np.mean(sempipes_scores)
sempipes_std = np.std(sempipes_scores)

print(f"{dataset_name},{sempipes_mean - baseline_mean},{baseline_mean},{baseline_std},{sempipes_mean},{sempipes_std}")
