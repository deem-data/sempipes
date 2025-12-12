import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

BRAND_WORDS = [
    "sony",
    "samsung",
    "panasonic",
    "toshiba",
    "sandisk",
    "intel",
    "lenovo",
    "hp",
    "dell",
    "apple",
    "asus",
    "acer",
    "philips",
    "msi",
    "lg",
    "seagate",
    "kingston",
    "transcend",
    "fujitsu",
    "western",
    "adata",
    "canon",
    "nikon",
    "olympus",
    "casio",
    "sharp",
    "jvc",
    "benq",
    "gateway",
    "nec",
    "siemens",
]
BRAND_RX = re.compile(r"\b(" + "|".join(BRAND_WORDS) + r")\b", re.IGNORECASE)
SKU_RX = re.compile(r"\b([a-z]{0,4}\d{2,}[a-z\d-]{0,}|[a-z]{2,8}\d{1,6})\b", re.I)
NUM_RX = re.compile(r"\b\d{3,}\b")


def char_ngrams(text, n=3):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def word_ngrams(text, n=2):
    tokens = str(text).lower().split()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else set()


def main_feats(text):
    text = str(text)
    feats = set(BRAND_RX.findall(text))
    feats.update(SKU_RX.findall(text))
    feats.update(NUM_RX.findall(text))
    return {f.lower() for f in feats if f}


def extract_features(row, cols, use_category=False):
    values = [str(row[c]) for c in cols if c in row and pd.notna(row[c])]
    combined = " ".join(values)
    feats = set()
    feats.update(main_feats(combined))
    feats.update(word_ngrams(combined, 2))
    feats.update(word_ngrams(combined, 3))
    feats.update(char_ngrams(combined, 3))
    if use_category and "category" in row:
        feats.add(str(row["category"]).strip().lower())
    feats.update({w for w in combined.lower().split() if len(w) > 1})
    return feats


def get_candidate_pairs(df, max_pairs, cols, use_category=False, block_sample=50):
    # Extract all features for each row
    all_feats = []
    id_array = df["id"].to_numpy()
    for row in tqdm(df.to_dict(orient="records"), desc="Feature Extraction"):
        all_feats.append(extract_features(row, cols, use_category=use_category))
    # Block building
    buckets = defaultdict(list)
    for idx, feats in enumerate(all_feats):
        for f in feats:
            buckets[f].append(id_array[idx])
    pairs_set = set()
    for block_key, ids in tqdm(buckets.items(), desc="Processing blocks"):
        ids = list(set(ids))
        if 2 <= len(ids) <= block_sample:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    if a < b:
                        pairs_set.add((a, b))
                    else:
                        pairs_set.add((b, a))
                    if len(pairs_set) >= max_pairs:
                        break
                if len(pairs_set) >= max_pairs:
                    break
        elif len(ids) > block_sample:
            sampled = np.random.choice(ids, block_sample, replace=False)
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    a, b = sampled[i], sampled[j]
                    if a < b:
                        pairs_set.add((a, b))
                    else:
                        pairs_set.add((b, a))
                    if len(pairs_set) >= max_pairs:
                        break
                if len(pairs_set) >= max_pairs:
                    break
        if len(pairs_set) >= max_pairs:
            break
    pairs = list(pairs_set)[:max_pairs]
    # Pad if needed
    if len(pairs) < max_pairs:
        pairs.extend([(0, 0)] * (max_pairs - len(pairs)))
    return pairs


def save_output(X1_candidate_pairs, X2_candidate_pairs):
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
    all_pairs = X1_candidate_pairs + X2_candidate_pairs
    df_out = pd.DataFrame(all_pairs, columns=["left_instance_id", "right_instance_id"])
    df_out.to_csv("output_swe.csv", index=False)


if __name__ == "__main__":
    X1 = pd.read_csv("experiments/sigmod/hidden_data/Z1.csv")
    X2 = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv")
    # For X1: only 'title'
    X1_pairs = get_candidate_pairs(X1, 1000000, cols=["title"], block_sample=50)
    # For X2: all available fields, including category and description
    X2_pairs = get_candidate_pairs(
        X2, 2000000, cols=["name", "brand", "category", "description", "price"], use_category=True, block_sample=50
    )
    save_output(X1_pairs, X2_pairs)
