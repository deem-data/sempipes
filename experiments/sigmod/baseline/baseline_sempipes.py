import os
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import skrub
from tqdm import tqdm

import sempipes
from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label


def extract_blocking_features(data_ref: skrub.DataOp, text_column: str, name: str, how_many: int = 8) -> pd.DataFrame:
    """
    Generate blocking-relevant features from text using sem_gen_features.
    This automatically discovers features useful for blocking and entity matching.
    """
    data_ref = data_ref.skb.set_description(
        f"This is a dataset with text descriptions in the '{text_column}' column. "
        "The goal is to generate features useful for entity blocking and matching."
    )

    # Generate blocking features automatically using sem_gen_features
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

    # Evaluate the DataOp to get the DataFrame
    extracted = data_ref.skb.eval()

    # Ensure all generated feature columns are properly formatted
    for col in extracted.columns:
        if col not in ["id", text_column]:  # Skip id and original text column
            extracted[col] = extracted[col].fillna("0").astype(str).str.lower()

    return extracted


def block_with_extracted_features(X, text_attr, blocking_features):
    """
    Perform blocking using generated features for better performance.
    :param X: dataframe with original data
    :param text_attr: original text attribute name
    :param blocking_features: dataframe with generated blocking features
    :return: candidate set of tuple pairs
    """
    # Get all feature columns (exclude id and original text column)
    feature_cols = [col for col in blocking_features.columns if col not in ["id", text_attr]]

    # Build blocking indices from all generated features
    pattern2id_by_feature = {col: defaultdict(list) for col in feature_cols}

    # Also keep original pattern matching for fallback
    pattern2id_original_1 = defaultdict(list)
    pattern2id_original_2 = defaultdict(list)

    for i in tqdm(range(X.shape[0]), desc="Building blocking indices"):
        # Use all generated features for blocking
        for col in feature_cols:
            feature_value = str(blocking_features[col].iloc[i]).lower()
            if feature_value and feature_value != "0":
                # For string features, split by space and create blocks for each token
                tokens = feature_value.split()
                for token in tokens:
                    if len(token) > 1:  # Skip single characters
                        pattern2id_by_feature[col][token].append(i)
                # Also block on the full feature value if it's not too long
                if len(feature_value) < 200:  # Skip very long feature values
                    pattern2id_by_feature[col][feature_value].append(i)

        # Original blocking patterns (fallback)
        attr_i = str(X[text_attr][i])
        pattern_1 = attr_i.lower()
        pattern2id_original_1[pattern_1].append(i)

        pattern_2 = re.findall(r"\w+\s\w+\d+", attr_i)
        if len(pattern_2) > 0:
            pattern_2 = list(sorted(pattern_2))
            pattern_2 = [str(it).lower() for it in pattern_2]
            pattern2id_original_2[" ".join(pattern_2)].append(i)

    # Collect candidate pairs from all blocking strategies
    candidate_pairs_set = set()

    # Block on each generated feature
    for col in feature_cols:
        for pattern in tqdm(pattern2id_by_feature[col], desc=f"Blocking on {col}"):
            ids = list(sorted(pattern2id_by_feature[col][pattern]))
            # Adjust threshold based on feature type - more specific features can have higher thresholds
            threshold = 500 if len(pattern) > 10 else 200  # Longer patterns are more specific
            if len(ids) < threshold:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        candidate_pairs_set.add((ids[i], ids[j]))

    # Original blocking patterns
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

    # Sort candidate pairs by jaccard similarity for ranking
    jaccard_similarities = []
    candidate_pairs_real_ids = []
    for it in tqdm(candidate_pairs, desc="Computing similarities"):
        id1, id2 = it

        # Get real ids
        real_id1 = X["id"][id1]
        real_id2 = X["id"][id2]
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        # Compute jaccard similarity using all generated features
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
            # Fallback to original text
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

    # build index from patterns to tuples
    pattern2id_1 = defaultdict(list)
    pattern2id_2 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        pattern_1 = attr_i.lower()  # use the whole attribute as the pattern
        pattern2id_1[pattern_1].append(i)

        pattern_2 = re.findall("\w+\s\w+\d+", attr_i)  # look for patterns like "thinkpad x1"
        if len(pattern_2) == 0:
            continue
        pattern_2 = list(sorted(pattern_2))
        pattern_2 = [str(it).lower() for it in pattern_2]
        pattern2id_2[" ".join(pattern_2)].append(i)

    # add id pairs that share the same pattern to candidate set
    candidate_pairs_1 = []
    for pattern in tqdm(pattern2id_1):
        ids = list(sorted(pattern2id_1[pattern]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidate_pairs_1.append((ids[i], ids[j]))  #
    # add id pairs that share the same pattern to candidate set
    candidate_pairs_2 = []
    for pattern in tqdm(pattern2id_2):
        ids = list(sorted(pattern2id_2[pattern]))
        if len(ids) < 100:  # skip patterns that are too common
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_2.append((ids[i], ids[j]))

    # remove duplicate pairs and take union
    candidate_pairs = set(candidate_pairs_2)
    candidate_pairs = candidate_pairs.union(set(candidate_pairs_1))
    candidate_pairs = list(candidate_pairs)

    # sort candidate pairs by jaccard similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs
    jaccard_similarities = []
    candidate_pairs_real_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        # get real ids
        real_id1 = X["id"][id1]
        real_id2 = X["id"][id2]
        if (
            real_id1 < real_id2
        ):  # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        # compute jaccard similarity
        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids


def save_output(X1_candidate_pairs, X2_candidate_pairs, output_path="output_baseline.csv"):
    """Save the candidate set for both datasets to a SINGLE file"""
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # Make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # Make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv(output_path, index=False)


# Read the datasets
X1 = pd.read_csv("experiments/sigmod/hidden_data/Z1.csv")
X2 = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv")

print("Performing blocking for X1 (original method)...")
X1_candidate_pairs = block_with_attr(X1, attr="title")

num_repeats = 3  # Number of times to repeat the process
print(f"Repeating feature extraction and blocking for X2 {num_repeats} times...")

all_X2_candidate_pairs = []
X2_recalls = []  # Store recall for each iteration

base_path = "experiments/sigmod/hidden_data"
evaluation_dataset_path_Y2 = os.path.join(base_path, "Y2.csv")
output_path = "output_baseline.csv"

for repeat_idx in range(num_repeats):
    print(f"\n--- Repeat {repeat_idx + 1}/{num_repeats} ---")

    # Generate blocking features using sem_gen_features
    print(f"Generating blocking features for X2 (repeat {repeat_idx + 1})...")
    X2_dataop = skrub.var("X2", X2).skb.set_name("X2")
    X2_features = extract_blocking_features(
        X2_dataop, "name", f"generate_x2_blocking_features_repeat_{repeat_idx}", how_many=8
    )

    # Perform blocking for X2 using generated features
    print(f"Performing blocking for X2 (repeat {repeat_idx + 1})...")
    X2_candidate_pairs = block_with_extracted_features(X2, "name", X2_features)

    # Collect candidate pairs from this repeat
    all_X2_candidate_pairs.extend(X2_candidate_pairs)
    print(f"Found {len(X2_candidate_pairs)} candidate pairs in repeat {repeat_idx + 1}")

    # Save output and evaluate X2 recall for this iteration
    save_output(X1_candidate_pairs, X2_candidate_pairs, output_path)

    # Read from output and evaluate X2
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
