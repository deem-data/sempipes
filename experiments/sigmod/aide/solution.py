import csv
import os
import random
import re

import numpy as np
import pandas as pd

from experiments.sigmod.evaluation import calculate_metrics, get_evaluation_dataset_with_predicted_label


def normalize_text(s):
    s = s.lower()
    s = re.sub("[^a-z0-9]", " ", s)
    return s.strip()


def reservoir_sample(stream, k, seed=42):
    random.seed(seed)
    res = []
    for i, item in enumerate(stream):
        if i < k:
            res.append(item)
        else:
            j = random.randrange(i + 1)
            if j < k:
                res[j] = item
    return res


def generate_z1_pairs(ids, w=3):
    n = len(ids)
    for i in range(n):
        a = ids[i]
        for j in range(1, w + 1):
            if i + j < n:
                b = ids[i + j]
                yield (min(a, b), max(a, b))


def generate_z2_pairs(brands_ids, names_dict, w=5):
    # brands_ids: dict brand -> list of ids
    for brand, id_list in brands_ids.items():
        names = names_dict[brand]
        # sort by normalized name
        sorted_idx = sorted(range(len(id_list)), key=lambda i: names[i])
        n = len(sorted_idx)
        for idx in range(n):
            a = id_list[sorted_idx[idx]]
            for j in range(1, w + 1):
                if idx + j < n:
                    b = id_list[sorted_idx[idx + j]]
                    yield (min(a, b), max(a, b))


def save_output(X1_candidate_pairs, X2_candidate_pairs, output_path="working/submission.csv"):
    """Save the candidate set for both datasets to a SINGLE file"""
    expected_cand_size_X1 = 1_000_000
    expected_cand_size_X2 = 2_000_000

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


def compute_z1_blocking():
    """Compute blocking for Z1 dataset"""
    z1 = pd.read_csv("experiments/sigmod/hidden_data/Z1.csv", usecols=["id", "title"])
    z1["title_n"] = z1["title"].fillna("").map(normalize_text)
    z1 = z1.sort_values("title_n")
    ids1 = z1["id"].tolist()
    stream1 = generate_z1_pairs(ids1, w=3)
    cand1 = reservoir_sample(stream1, 1_000_000, seed=0)
    return cand1


def compute_z2_blocking(seed=1):
    """Compute blocking for Z2 dataset"""
    z2 = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv", usecols=["id", "brand", "name"])
    z2["brand_n"] = z2["brand"].fillna("").map(normalize_text)
    z2["name_n"] = z2["name"].fillna("").map(normalize_text)
    brands = {}
    names = {}
    for b, grp in z2.groupby("brand_n"):
        ids = grp["id"].tolist()
        nm = grp["name_n"].tolist()
        if len(ids) > 1:
            brands[b] = ids
            names[b] = nm
    stream2 = generate_z2_pairs(brands, names, w=5)
    cand2 = reservoir_sample(stream2, 2_000_000, seed=seed)
    return cand2


if __name__ == "__main__":
    print("Performing blocking for Z1 (original method)...")
    cand1 = compute_z1_blocking()
    print(f"Found {len(cand1)} candidate pairs for Z1")

    num_repeats = 5
    print(f"Repeating blocking for Z2 {num_repeats} times...")

    all_cand2 = []
    Z2_recalls = []

    base_path = "experiments/sigmod/hidden_data"
    evaluation_dataset_path_Y2 = os.path.join(base_path, "Y2.csv")
    output_path = "experiments/sigmod/results/output_aide.csv"

    for repeat_idx in range(num_repeats):
        print(f"\n--- Repeat {repeat_idx + 1}/{num_repeats} ---")

        # Perform blocking for Z2 (with different seed for each repeat)
        print(f"Performing blocking for Z2 (repeat {repeat_idx + 1})...")
        cand2 = compute_z2_blocking(seed=1 + repeat_idx)

        # Collect candidate pairs from this repeat
        all_cand2.extend(cand2)
        print(f"Found {len(cand2)} candidate pairs in repeat {repeat_idx + 1}")

        # Save output and evaluate Z2 recall for this iteration
        save_output(cand1, cand2, output_path)

        # Read from output and evaluate Z2
        evaluation_dataset, submission_dataset = get_evaluation_dataset_with_predicted_label(
            evaluation_dataset_path_Y2, output_path, dataset_id=2
        )

        recall, tp, all = calculate_metrics(evaluation_dataset, submission_dataset)
        Z2_recalls.append(recall)
        print(f"Recall for Y2.csv (repeat {repeat_idx + 1}) is {recall}.")

    avg_recall = np.mean(Z2_recalls)
    std_recall = np.std(Z2_recalls)
    print(f"\nAverage recall: {round(avg_recall, 3)}")
    print(f"Standard deviation: {round(std_recall, 3)}")
