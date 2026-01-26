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
MODEL_RX = re.compile(r"\b([a-zA-Z]{0,4}\d{2,}[a-zA-Z0-9\-]{0,})\b")
CAPACITY_RX = re.compile(r"\b(\d{1,5}\s?(?:gb|tb|mb|mhz|ghz|w|inch|in))\b", re.IGNORECASE)
VERSION_RX = re.compile(r"\b(v?\d+\.\d+|mark\s*[ivx]+|gen(?:eration)?\s*\d+)\b", re.IGNORECASE)
DIGIT_WORD_RX = re.compile(r"\b\d+\b")


def regex_features(text):
    features = set()
    text = str(text)
    # Brand matches
    features.update(m.group(1).lower() for m in BRAND_RX.finditer(text))
    # Model/SKU-like tokens (e.g. BX807151380, 13T, X540U etc)
    features.update(m.group(1).lower() for m in MODEL_RX.finditer(text))
    # Capacity/spec (e.g. 64GB, 512MB, 2TB, 2400MHZ)
    features.update(m.group(1).lower() for m in CAPACITY_RX.finditer(text))
    # Version/Mark (e.g. v2.0, mark iv, generation 3)
    features.update(m.group(1).lower() for m in VERSION_RX.finditer(text))
    # Pure digit tokens
    features.update(m.group(0) for m in DIGIT_WORD_RX.finditer(text))
    # 2-3 token ngrams using regex
    tokens = [t.group(0).lower() for t in re.finditer(r"\b[a-z0-9]+\b", text)]
    for i in range(len(tokens) - 1):
        features.add(tokens[i] + " " + tokens[i + 1])
        if i < len(tokens) - 2:
            features.add(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])
    return features


def extract_features_X1(row):
    title = str(row["title"])
    feats = regex_features(title)
    return feats


def extract_features_X2(row):
    name = str(row.get("name", ""))
    brand = str(row.get("brand", ""))
    category = str(row.get("category", ""))
    descr = str(row.get("description", ""))
    # Concatenate all text fields for regex extraction
    text = " ".join([name, brand, category, descr])
    feats = regex_features(text)
    # Optionally, bin price
    if "price" in row and not pd.isnull(row["price"]):
        try:
            price = float(row["price"])
            feats.add("pricebin_" + str(int(price // 10)))
            feats.add("pricebin_" + str(int(price // 25)))
        except Exception:
            pass
    return feats


def get_random_unique_pairs(ids, existing_pairs_set, needed):
    output = set()
    n = len(ids)
    if n < 2 or needed <= 0:
        return []
    rng = np.random.default_rng()
    trials = 0
    batchsize = min(needed * 2, 1000000)
    while len(output) < needed and trials < 1000:
        idx_a = rng.integers(0, n, size=batchsize)
        idx_b = rng.integers(0, n, size=batchsize)
        mask = idx_a != idx_b
        idx_a = idx_a[mask]
        idx_b = idx_b[mask]
        min_ab = np.minimum(ids[idx_a], ids[idx_b])
        max_ab = np.maximum(ids[idx_a], ids[idx_b])
        pairs = zip(min_ab, max_ab)
        for tup in pairs:
            if tup in existing_pairs_set or tup in output:
                continue
            output.add(tup)
            if len(output) >= needed:
                break
        trials += 1
    return list(output)


def block_candidates(df, feats_fn, max_pairs, random_fill=True):
    buckets = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        patterns = feats_fn(row)
        for p in patterns:
            if p:
                buckets[p].append(row["id"])
    pairs_set = set()
    for id_list in tqdm(buckets.values()):
        id_list = sorted(set(id_list))
        if len(id_list) > 50:
            id_list = list(np.random.default_rng().choice(id_list, 300, replace=True))
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                a, b = id_list[i], id_list[j]
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pairs_set.add((a, b))
                if len(pairs_set) >= 500000:
                    break
            if len(pairs_set) >= 500000:
                break
        if len(pairs_set) >= 500000:
            break
    pairs = list(pairs_set)
    # forcibly fill quota with random unique if needed
    if len(pairs) < max_pairs and random_fill:
        ids = df["id"].drop_duplicates().to_numpy()
        needed = max_pairs - len(pairs)
        random_pairs = get_random_unique_pairs(ids, pairs_set, needed)
        pairs += random_pairs
    pairs = pairs[:max_pairs]
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
    output_df = pd.DataFrame(all_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


def main():
    X1 = pd.read_csv("Z1.csv")
    X1["id"] = X1["id"].astype(int)
    X2 = pd.read_csv("Z2.csv")
    X2["id"] = X2["id"].astype(int)
    X1_candidate_pairs = block_candidates(X1, extract_features_X1, max_pairs=1000000)
    X2_candidate_pairs = block_candidates(X2, extract_features_X2, max_pairs=2000000)
    save_output(X1_candidate_pairs, X2_candidate_pairs)


if __name__ == "__main__":
    main()
