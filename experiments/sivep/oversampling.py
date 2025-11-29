import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
all_data = pd.read_csv("experiments/sivep/data.csv")

scores = []
for seed in [42, 1337, 2025, 7321, 98765]:
    df = all_data.sample(frac=0.1, random_state=seed)

    # Remove records with unreported race
    df = df[df.cs_raca != 9]
    # Remove influenza cases
    df = df[~df.classi_fin.isin([1])]
    # Target label: SRAG due to covid
    df["due_to_covid"] = df.classi_fin == 5

    data = df.drop(columns=["classi_fin", "evolucao", "vacina_cov", "cs_sexo", "dt_evoluca", "dt_interna"])
    train, test = train_test_split(data, test_size=0.1, random_state=seed)

    train_indigenous = train[train.cs_raca == 5].copy(deep=True)
    num_extra_samples = 2 * len(train_indigenous)
    extra_samples = train_indigenous.sample(n=num_extra_samples, replace=True, random_state=seed)
    augmented_train = pd.concat([train, extra_samples], ignore_index=True)
    augmented_train_labels = augmented_train.due_to_covid
    augmented_train = augmented_train.drop(columns=["due_to_covid"])

    train_labels = train.due_to_covid
    train = train.drop(columns=["due_to_covid"])

    model = XGBClassifier(eval_metric="logloss", random_state=seed)
    model.fit(train, train_labels)

    augmented_model = XGBClassifier(eval_metric="logloss", random_state=seed)
    augmented_model.fit(augmented_train, augmented_train_labels)

    majority_groups = {1, 2, 3, 4}

    test_minority = test[~test.cs_raca.isin(majority_groups)]

    test_minority_labels = test_minority.due_to_covid
    test_minority = test_minority.drop(columns=["due_to_covid"])

    minority_score = roc_auc_score(test_minority_labels, model.predict_proba(test_minority)[:, 1])
    augmented_minority_score = roc_auc_score(test_minority_labels, augmented_model.predict_proba(test_minority)[:, 1])

    print(f"ROC AUC score for minority group on seed {seed}: {minority_score} -> {augmented_minority_score}")
    scores.append(augmented_minority_score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
