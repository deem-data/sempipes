import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
all_data = pd.read_csv("experiments/sivep/data.csv")

non_augmented_scores = []
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
    data = data.drop(columns=["sem_pri", "sem_not"])  # These columns must be dropped as well due to type errors
    train, test = train_test_split(data, test_size=0.5, random_state=seed)
    train_labels = train.due_to_covid

    ###
    # --- Data augmentation for indigenous group (cs_raca==5) ---
    # Oversample indigenous group in training set to improve fairness
    indigenous_train = train[train.cs_raca == 5]
    if len(indigenous_train) > 0:
        # Oversample by duplicating indigenous samples to match the majority class size
        n_majority = train.shape[0] - indigenous_train.shape[0]
        n_indigenous = indigenous_train.shape[0]
        n_repeat = max(1, n_majority // max(1, n_indigenous))
        train_aug = pd.concat([train] + [indigenous_train]*n_repeat, ignore_index=True)
        train_labels_aug = pd.concat([train["due_to_covid"]] + [indigenous_train["due_to_covid"]]*n_repeat, ignore_index=True)
        train = train_aug
        train_labels = train_labels_aug
    ###
    train = train.drop(columns=["due_to_covid"])

    model = XGBClassifier(eval_metric="logloss", random_state=seed)
    model.fit(train, train_labels)

    majority_groups = {1, 2, 3, 4}
    test_minority = test[~test.cs_raca.isin(majority_groups)]
    test_minority_labels = test_minority.due_to_covid
    test_minority = test_minority.drop(columns=["due_to_covid"])

    minority_score = roc_auc_score(test_minority_labels, model.predict_proba(test_minority)[:, 1])

    print(f"ROC AUC score for minority group on seed {seed}: {minority_score}")
    scores.append(minority_score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
