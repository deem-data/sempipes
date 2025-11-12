# from tabpfn import TabPFNClassifier
# from tabpfn.constants import ModelVersion
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from experiments.caafe import scoring

warnings.filterwarnings("ignore")

datasets = ["airlines", "balance-scale", "tic-tac-toe"]

prompt_id = "v3"

for dataset_name in datasets:
    baselines_scores = []
    caafe_scores = []

    failures = 0
    for seed in range(0, 5):
        data = pd.read_csv(f"experiments/caafe/data/{dataset_name}/data.csv")
        target_column = data.columns[-1]

        train, test = train_test_split(data, test_size=0.25, random_state=seed)

        y_train = train[target_column]
        X_train = train.drop(columns=[target_column])
        X_train = X_train.fillna(0)

        y_test = test[target_column]
        X_test = test.drop(columns=[target_column])
        X_test = X_test.fillna(0)

        X_train_baseline = X_train.copy(deep=True)
        X_test_baseline = X_test.copy(deep=True)

        baseline_clf = RandomForestClassifier(random_state=seed)
        # baseline_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        baseline_clf.fit(X_train_baseline, y_train)

        baseline_y_pred = baseline_clf.predict_proba(X_test_baseline)
        baseline_score = scoring(y_test, baseline_y_pred)
        baselines_scores.append(baseline_score)

        with open(
            f"experiments/caafe/data/{dataset_name}/{dataset_name}_{prompt_id}_{seed}_code.txt", "r", encoding="utf-8"
        ) as f:
            code = f.read()

        def gen_features(code, data):
            local_vars = {"df": data.copy()}
            exec(code, {}, local_vars)
            return local_vars["df"]

        try:
            X_train_caafe = gen_features(code, X_train)
            X_train_caafe = X_train_caafe.replace([np.inf, -np.inf, np.nan], 0)
            X_test_caafe = gen_features(code, X_test)
            X_test_caafe = X_test_caafe.replace([np.inf, -np.inf, np.nan], 0)

            caafe_clf = RandomForestClassifier(random_state=seed)
            # caafe_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
            caafe_clf.fit(X_train_caafe, y_train)

            caafe_y_pred = caafe_clf.predict_proba(X_test_caafe)
            caafe_score = scoring(y_test, caafe_y_pred)

            # print('#', dataset_name, seed, prompt_id, baseline_score, caafe_score)
            caafe_scores.append(caafe_score)

        except Exception:
            failures += 1
            # pass
            # print(f"⚠️ Exception occurred in {dataset_name} {seed}: {e}")
            # import traceback
            # traceback.print_exc()

    baseline_mean = np.mean(baselines_scores)
    baseline_std = np.std(baselines_scores)
    caafe_mean = np.mean(caafe_scores)
    caafe_std = np.std(caafe_scores)

    print(
        f"{dataset_name},{caafe_mean - baseline_mean},{baseline_mean},{baseline_std},{caafe_mean},{caafe_std},{failures}"
    )
