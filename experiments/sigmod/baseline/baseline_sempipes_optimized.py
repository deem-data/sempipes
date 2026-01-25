import os
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator
from tqdm import tqdm

import sempipes
from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label
from sempipes.optimisers import EvolutionarySearch, optimise_colopro


def extract_blocking_features_sempipes(
    data_ref: skrub.DataOp, text_column: str, name: str, how_many: int = 8
) -> skrub.DataOp:
    """
    Generate blocking-relevant features from text using sem_gen_features.
    This automatically discovers features useful for blocking and entity matching.
    """
    data_ref = data_ref.skb.set_description(
        f"This is a dataset with text descriptions in the '{text_column}' column. "
        "The goal is to generate features useful for entity blocking and matching."
    )

    data_ref = data_ref.sem_gen_features(
        nl_prompt=(
            "You are analyzing product/item descriptions for entity blocking and matching. "
            "The goal is to find duplicate or matching items that may be described differently. "
            f"The dataset contains text in the '{text_column}' column. "
            "\n\nTASK: Generate features that are useful for blocking and matching similar items. "
            "Focus on features that can help identify when two items are the same or very similar, "
            "even if they are described differently (different languages, formatting, typos, etc.). "
            "\n\nFEATURE REQUIREMENTS:\n"
            "1. Features should be commonly present (>10% of rows) to be useful for blocking. "
            "2. Features should be language-independent when possible (normalize multilingual variations). "
            "3. Features should be extractable using regex and rule-based methods (NO transformers/LLMs). "
            "4. Features should help distinguish between different items while matching similar ones. "
            "\n\nSUGGESTED FEATURE TYPES:\n"
            "- Normalized/cleaned text versions (lowercase, whitespace normalized, special chars handled). "
            "- Key words/tokens extracted from text (brand names, model identifiers, product types). "
            "- Key phrases (2-3 word sequences like 'thinkpad x1', 'usb 3.0'). "
            "- Model numbers, SKUs, alphanumeric codes. "
            "- Numeric sequences (capacities, sizes, etc.). "
            "- Normalized variations (e.g., 'san disk' -> 'sandisk', 'micro sd' -> 'microsd'). "
            "\n\nCRITICAL FOR RECALL: Generate features that maximize recall. "
            "It's better to have features that catch potential matches (even with some false positives) "
            "than to miss matches. Missing a feature means missing potential matches. "
            "\n\nGenerate Python code using regex (re module) and basic string operations. "
            "Return features as lowercase strings, '0' for missing values. "
            "Optimize for throughput - must handle millions of rows efficiently."
        ),
        name=name,
        how_many=how_many,
    )

    def fix_up(df):
        for col in df.columns:
            if col not in ["id", text_column]:
                df[col] = df[col].fillna("0").astype(str).str.lower()
        return df

    return data_ref.skb.apply_func(fix_up)


def block_with_extracted_features(X, text_attr, blocking_features):
    """
    Perform blocking using generated features for better performance.
    :param X: dataframe with original data
    :param text_attr: original text attribute name
    :param blocking_features: dataframe with generated blocking features
    :return: candidate set of tuple pairs
    """
    feature_cols = [col for col in blocking_features.columns if col not in ["id", text_attr]]

    pattern2id_by_feature = {col: defaultdict(list) for col in feature_cols}
    pattern2id_original_1 = defaultdict(list)
    pattern2id_original_2 = defaultdict(list)

    for i in tqdm(range(X.shape[0]), desc="Building blocking indices"):
        for col in feature_cols:
            feature_value = str(blocking_features[col].iloc[i]).lower()
            if feature_value and feature_value != "0":
                tokens = feature_value.split()
                for token in tokens:
                    if len(token) > 1:
                        pattern2id_by_feature[col][token].append(i)
                if len(feature_value) < 200:
                    pattern2id_by_feature[col][feature_value].append(i)

        attr_i = str(X[text_attr][i])
        pattern_1 = attr_i.lower()
        pattern2id_original_1[pattern_1].append(i)

        pattern_2 = re.findall(r"\w+\s\w+\d+", attr_i)
        if len(pattern_2) > 0:
            pattern_2 = list(sorted(pattern_2))
            pattern_2 = [str(it).lower() for it in pattern_2]
            pattern2id_original_2[" ".join(pattern_2)].append(i)

    candidate_pairs_set = set()

    for col in feature_cols:
        for pattern in tqdm(pattern2id_by_feature[col], desc=f"Blocking on {col}"):
            ids = list(sorted(pattern2id_by_feature[col][pattern]))
            threshold = 500 if len(pattern) > 10 else 200
            if len(ids) < threshold:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        candidate_pairs_set.add((ids[i], ids[j]))

    for pattern in tqdm(pattern2id_original_1, desc="Blocking on original patterns"):
        ids = list(sorted(pattern2id_original_1[pattern]))
        if len(ids) < 1000:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_set.add((ids[i], ids[j]))

    for pattern in tqdm(pattern2id_original_2, desc="Blocking on original pattern 2"):
        ids = list(sorted(pattern2id_original_2[pattern]))
        if len(ids) < 100:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_set.add((ids[i], ids[j]))

    candidate_pairs = list(candidate_pairs_set)

    jaccard_similarities = []
    candidate_pairs_real_ids = []

    for it in tqdm(candidate_pairs, desc="Computing similarities"):
        id1, id2 = it

        real_id1 = X["id"][id1]
        real_id2 = X["id"][id2]
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        all_tokens1 = set()
        all_tokens2 = set()
        for col in feature_cols:
            val1 = str(blocking_features[col].iloc[id1]).lower()
            val2 = str(blocking_features[col].iloc[id2]).lower()
            if val1 and val1 != "0":
                all_tokens1.update(val1.split())
            if val2 and val2 != "0":
                all_tokens2.update(val2.split())

        if len(all_tokens1) == 0 or len(all_tokens2) == 0:
            name1 = str(X[text_attr][id1])
            name2 = str(X[text_attr][id2])
            all_tokens1 = set(name1.lower().split())
            all_tokens2 = set(name2.lower().split())

        if len(all_tokens1) > 0 or len(all_tokens2) > 0:
            jaccard = len(all_tokens1.intersection(all_tokens2)) / max(len(all_tokens1), len(all_tokens2))
        else:
            jaccard = 0.0
        jaccard_similarities.append(jaccard)

    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids


def block_with_attr(X, attr):
    """
    This function performs blocking using attr (original baseline method).
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """

    pattern2id_1 = defaultdict(list)
    pattern2id_2 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        pattern_1 = attr_i.lower()
        pattern2id_1[pattern_1].append(i)

        pattern_2 = re.findall("\w+\s\w+\d+", attr_i)
        if len(pattern_2) == 0:
            continue
        pattern_2 = list(sorted(pattern_2))
        pattern_2 = [str(it).lower() for it in pattern_2]
        pattern2id_2[" ".join(pattern_2)].append(i)

    candidate_pairs_1 = []
    for pattern in tqdm(pattern2id_1):
        ids = list(sorted(pattern2id_1[pattern]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidate_pairs_1.append((ids[i], ids[j]))
    candidate_pairs_2 = []
    for pattern in tqdm(pattern2id_2):
        ids = list(sorted(pattern2id_2[pattern]))
        if len(ids) < 100:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_2.append((ids[i], ids[j]))

    candidate_pairs = set(candidate_pairs_2)
    candidate_pairs = candidate_pairs.union(set(candidate_pairs_1))
    candidate_pairs = list(candidate_pairs)

    jaccard_similarities = []
    candidate_pairs_real_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        real_id1 = X["id"][id1]
        real_id2 = X["id"][id2]
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids


class BlockingModel(BaseEstimator):
    def __init__(self, size_of_output=2000000):
        self.size_of_output = size_of_output

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, dict):
            X_data = X.get("_skrub_X", X)
        else:
            X_data = X

        if not isinstance(X_data, pd.DataFrame):
            X_data = pd.DataFrame(X_data)

        X_data = X_data.reset_index(drop=True)

        text_attr = "name"

        # Separate original data from blocking features to match original logic
        # X: original data with 'id' and 'name'
        # blocking_features: full dataframe with 'id', 'name', and generated feature columns
        X_original = X_data[["id", text_attr]].copy()
        blocking_features = X_data.copy()

        return block_with_extracted_features(X_original, text_attr, blocking_features)


def calculate_recall(estimator, X, y):
    """
    Sklearn-compatible scorer function that calculates recall.

    Args:
        estimator: The fitted estimator (pipeline)
        X: Test features (DataFrame or dict with '_skrub_X' key)
        y: Test labels (DataFrame with 'lid' and 'rid' columns from skrub's cross_validate)

    Returns:
        float: Recall score
    """
    predictions = estimator.predict(X)
    if isinstance(predictions, list):
        predicted_df = pd.DataFrame(predictions, columns=["left_instance_id", "right_instance_id"])
    elif isinstance(predictions, pd.DataFrame):
        predicted_df = predictions.copy()
    else:
        predicted_df = pd.DataFrame(predictions)
        if "left_instance_id" not in predicted_df.columns or "right_instance_id" not in predicted_df.columns:
            if len(predicted_df.columns) == 2:
                predicted_df.columns = ["left_instance_id", "right_instance_id"]
            else:
                raise ValueError(f"Unexpected prediction format: {type(predictions)}")

    if not isinstance(y, pd.DataFrame):
        ground_truth = pd.DataFrame(y)
    else:
        ground_truth = y.copy()

    if "lid" not in ground_truth.columns or "rid" not in ground_truth.columns:
        raise ValueError(f"Ground truth labels must have 'lid' and 'rid' columns. Got: {ground_truth.columns.tolist()}")

    predicted_df["left_right"] = predicted_df["left_instance_id"].astype(str) + predicted_df[
        "right_instance_id"
    ].astype(str)
    predicted_values = predicted_df["left_right"].values

    ground_truth["left_right"] = ground_truth["lid"].astype(str) + ground_truth["rid"].astype(str)
    reference_values = ground_truth["left_right"].values

    inter = set.intersection(set(predicted_values), set(reference_values))
    recall = len(inter) / len(reference_values) if len(reference_values) > 0 else 0.0

    return round(recall, 3)


def _pipeline(operator_name):
    data_ref = skrub.var("data_original_x2").skb.mark_as_X()
    y = skrub.var("dummy_y").skb.mark_as_y()

    data_ref = data_ref.skb.set_description(
        "This is a dataset with text descriptions in the 'name' column. "
        "The goal is to generate features useful for entity blocking and matching."
    )

    features = data_ref.sem_gen_features(
        nl_prompt=(
            "You are analyzing product/item descriptions for entity blocking and matching. "
            "The goal is to find duplicate or matching items that may be described differently. "
            "The dataset contains text in the 'name' column. "
            "\n\nTASK: Generate features that are useful for blocking and matching similar items. "
            "Focus on features that can help identify when two items are the same or very similar, "
            "even if they are described differently (different languages, formatting, typos, etc.). "
            "\n\nFEATURE REQUIREMENTS:\n"
            "1. Features should be commonly present (>10% of rows) to be useful for blocking. "
            "2. Features should be language-independent when possible (normalize multilingual variations). "
            "3. Features should be extractable using regex and rule-based methods (NO transformers/LLMs). "
            "4. Features should help distinguish between different items while matching similar ones. "
            "\n\nSUGGESTED FEATURE TYPES:\n"
            "- Normalized/cleaned text versions (lowercase, whitespace normalized, special chars handled). "
            "- Key words/tokens extracted from text (brand names, model identifiers, product types). "
            "- Key phrases (2-3 word sequences like 'thinkpad x1', 'usb 3.0'). "
            "- Model numbers, SKUs, alphanumeric codes. "
            "- Numeric sequences (capacities, sizes, etc.). "
            "- Normalized variations (e.g., 'san disk' -> 'sandisk', 'micro sd' -> 'microsd'). "
            "\n\nCRITICAL FOR RECALL: Generate features that maximize recall. "
            "It's better to have features that catch potential matches (even with some false positives) "
            "than to miss matches. Missing a feature means missing potential matches. "
            "\n\nGenerate Python code using regex (re module) and basic string operations. "
            "Return features as lowercase strings, '0' for missing values. "
            "Optimize for throughput - must handle millions of rows efficiently."
        ),
        name=operator_name,
        how_many=8,
    )

    def fix_up(df):
        for col in df.columns:
            if col not in ["id", "name"]:
                df[col] = df[col].fillna("0").astype(str).str.lower()
        return df

    features = features.skb.apply_func(fix_up)
    return features.skb.apply(BlockingModel(2000000), y=y)


def _create_env(X, y, operator_name, state):
    """Create environment dictionary for learner."""
    dummy_y = pd.Series([0] * len(X), name="dummy")

    return {
        "_skrub_X": X,
        "_skrub_y": dummy_y,
        "data_original_x2": X,
        "dummy_y": dummy_y,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_prefitted_state__{operator_name}": state,
        f"sempipes_inspirations__{operator_name}": None,
    }


def save_output(X1_candidate_pairs, X2_candidate_pairs, output_path="output_baseline.csv"):
    """Save the candidate set for both datasets to a SINGLE file"""
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv(output_path, index=False)


X1 = pd.read_csv("experiments/sigmod/hidden_data/Z1.csv")

print("Performing blocking for X1 (original method)...")
X1_candidate_pairs = pd.read_csv("baseine_output_X1.csv").values.tolist()

sample_labels = pd.read_csv("experiments/sigmod/data/Y2.csv")
train_X = pd.read_csv("experiments/sigmod/data/X2.csv")
train_labels = sample_labels.copy()

train_X["name"] = train_X["name"].str.lower()

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-3-flash-preview",
        parameters={"temperature": 2.0},
    ),
    llm_for_batch_processing=sempipes.LLM(
        name="gemini/gemini-3-flash-preview",
        parameters={"temperature": 2.0},
    ),
)

num_repeats = 3
print(f"Repeating feature extraction and blocking for X2 {num_repeats} times...")

all_X2_candidate_pairs = []
X2_recalls = []

base_path = "experiments/sigmod/hidden_data"
evaluation_dataset_path_Y2 = os.path.join(base_path, "Y2.csv")
output_path = "output_baseline.csv"

test_data = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv")
test_data["name"] = test_data["name"].str.lower()


def recall_scorer_with_labels(estimator, X_test, y=None, **kwargs):
    if isinstance(X_test, dict):
        X_test_data = X_test.get("_skrub_X", X_test)
    else:
        X_test_data = X_test
    if "id" in X_test_data.columns:
        test_ids = set(X_test_data["id"].values)
        test_labels = train_labels[train_labels["lid"].isin(test_ids) & train_labels["rid"].isin(test_ids)].copy()
        return calculate_recall(estimator, X_test, y=test_labels, **kwargs)
    else:
        return calculate_recall(estimator, X_test, y=train_labels, **kwargs)


for repeat_idx in range(num_repeats):
    print(f"\n--- Repeat {repeat_idx + 1}/{num_repeats} ---")

    operator_name = f"generate_x2_blocking_features_repeat_{repeat_idx}"
    pipeline_to_optimise = _pipeline(operator_name)

    print(f"Starting colopro optimization (repeat {repeat_idx + 1})...")
    outcomes = optimise_colopro(
        pipeline_to_optimise,
        operator_name,
        scoring=recall_scorer_with_labels,
        cv=5,
        num_trials=24,
        search=EvolutionarySearch(population_size=6),
        additional_env_variables={"data_original_x2": train_X, "dummy_y": pd.Series([0] * len(train_X), name="dummy")},
    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print(
        f"Best outcome score after optimization on train CV (repeat {repeat_idx + 1}): {best_outcome.score}, state: {best_outcome.state}"
    )

    learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
    learner_optimized.fit(_create_env(train_X, None, operator_name, best_outcome.state))
    X2_candidate_pairs = learner_optimized.predict(_create_env(test_data, None, operator_name, best_outcome.state))

    all_X2_candidate_pairs.extend(X2_candidate_pairs)
    print(f"Found {len(X2_candidate_pairs)} candidate pairs in repeat {repeat_idx + 1}")

    save_output(X1_candidate_pairs, X2_candidate_pairs, output_path)

    evaluation_dataset, submission_dataset = get_evaluation_dataset_with_predicted_label(
        evaluation_dataset_path_Y2, output_path, dataset_id=2
    )

    recall, tp, all = calculate_metrics(evaluation_dataset, submission_dataset)
    X2_recalls.append(recall)
    print(f"Recall for Y2.csv (repeat {repeat_idx + 1}) is {recall}.")

avg_recall = np.mean(X2_recalls)
std_recall = np.std(X2_recalls)
print(f"Average recall: {round(avg_recall, 3)}")
print(f"Standard deviation: {round(std_recall, 3)}")
