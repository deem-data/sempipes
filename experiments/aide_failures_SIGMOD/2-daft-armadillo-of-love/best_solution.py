import itertools
import random
import re
from collections import defaultdict

import pandas as pd


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def generate_candidates(df, text_col, budget, seed=42):
    random.seed(seed)
    # build inverted index
    inv = defaultdict(list)
    for idx, row in df.iterrows():
        rec_id = row["id"]
        tokens = normalize(row[text_col])
        for t in set(tokens):
            inv[t].append(rec_id)
    # exclude too common tokens
    max_df = max(2, int(len(df) * 0.1))
    candidates = set()
    for t, ids in inv.items():
        if 2 <= len(ids) <= max_df:
            for a, b in itertools.combinations(sorted(ids), 2):
                candidates.add((a, b))
        if len(candidates) >= budget:
            break
    # trim or pad
    cand_list = list(candidates)
    if len(cand_list) > budget:
        cand_list = random.sample(cand_list, budget)
    else:
        all_ids = df["id"].tolist()
        needed = budget - len(cand_list)
        existing = set(cand_list)
        while needed > 0:
            a, b = random.sample(all_ids, 2)
            if a > b:
                a, b = b, a
            if a != b and (a, b) not in existing:
                cand_list.append((a, b))
                existing.add((a, b))
                needed -= 1
    return cand_list


def recall_score(cands, truth):
    truth_set = set(tuple(x) for x in truth.values)
    hit = len([1 for pair in cands if pair in truth_set])
    return hit / len(truth_set)


def main():
    # load data
    X1 = pd.read_csv("input/X1.csv")
    Y1 = pd.read_csv("input/Y1.csv", names=["lid", "rid"], header=0)
    X2 = pd.read_csv("input/X2.csv")
    Y2 = pd.read_csv("input/Y2.csv", names=["lid", "rid"], header=0)
    # generate
    c1 = generate_candidates(X1, "title", 1_000_000, seed=42)
    # combine brand+name for X2
    X2["brand_name"] = X2["brand"].fillna("") + " " + X2["name"].fillna("")
    c2 = generate_candidates(X2, "brand_name", 2_000_000, seed=43)
    # recall
    rec1 = recall_score(c1, Y1)
    rec2 = recall_score(c2, Y2)
    overall = (rec1 * len(Y1) + rec2 * len(Y2)) / (len(Y1) + len(Y2))
    print(f"Recall X1: {rec1:.4f}, Recall X2: {rec2:.4f}, Overall Recall: {overall:.4f}")
    # save submission
    df_sub = pd.DataFrame(c1 + c2, columns=["left_instance_id", "right_instance_id"])
    df_sub.to_csv("working/submission.csv", index=False)


if __name__ == "__main__":
    main()
