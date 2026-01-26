import csv
import itertools
import os
import re

import pandas as pd


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def load_data(x_path, y_path, text_func):
    df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    # build ground-truth pairs set
    y_pairs = set()
    for a, b in zip(y_df.iloc[:, 0], y_df.iloc[:, 1]):
        i1, i2 = int(a), int(b)
        if i1 > i2:
            i1, i2 = i2, i1
        y_pairs.add((i1, i2))
    # build token sets
    tokens = {}
    for _, row in df.iterrows():
        idx = int(row["id"])
        text = text_func(row)
        tokens[idx] = tokenize(text if isinstance(text, str) else "")
    ids = sorted(tokens.keys())
    return ids, tokens, y_pairs


def gen_top_k(ids, tokens, k):
    sims = []
    for i, j in itertools.combinations(ids, 2):
        t1, t2 = tokens[i], tokens[j]
        inter = t1 & t2
        if not inter:
            score = 0.0
        else:
            score = len(inter) / len(t1 | t2)
        sims.append((i, j, score))
    sims.sort(key=lambda x: x[2], reverse=True)
    top = [(i, j) for i, j, _ in sims[:k]]
    return top


# Load and process X1
ids1, tokens1, y1 = load_data("input/X1.csv", "input/Y1.csv", lambda r: r["title"])
# Load and process X2 (concatenate name + brand)
ids2, tokens2, y2 = load_data(
    "input/X2.csv",
    "input/Y2.csv",
    lambda r: f"{r['name']} {r['brand']}" if pd.notna(r["brand"]) else r["name"],
)

K1, K2 = 1000000, 2000000
cands1 = gen_top_k(ids1, tokens1, K1)
cands2 = gen_top_k(ids2, tokens2, K2)

# Evaluate recall
rec1 = sum(1 for p in cands1 if p in y1) / len(y1) if y1 else 0.0
rec2 = sum(1 for p in cands2 if p in y2) / len(y2) if y2 else 0.0
overall = (rec1 * len(y1) + rec2 * len(y2)) / (len(y1) + len(y2)) if (y1 or y2) else 0.0
print(f"Recall X1: {rec1:.4f}")
print(f"Recall X2: {rec2:.4f}")
print(f"Overall Recall: {overall:.4f}")

# Write submission.csv
os.makedirs("working", exist_ok=True)
with open("working/submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["left_instance_id", "right_instance_id"])
    for i, j in cands1 + cands2:
        writer.writerow([i, j])
