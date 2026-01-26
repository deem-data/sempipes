import itertools
import os
import random
import re

import pandas as pd


def tokenize(s):
    if pd.isna(s):
        return []
    return [t for t in re.split(r"\W+", s.lower()) if len(t) > 2]


def generate_candidates(df, id_col, text_col, target, min_df=2, max_df=500):
    # build inverted index
    inv = {}
    for idx, row in df.iterrows():
        toks = set(tokenize(row[text_col]))
        for t in toks:
            inv.setdefault(t, []).append(row[id_col])
    # filter tokens
    toks = [t for t, ids in inv.items() if min_df <= len(ids) <= max_df]
    toks.sort(key=lambda t: len(inv[t]))
    candidates = set()
    for t in toks:
        ids = inv[t]
        rem = target - len(candidates)
        if rem <= 0:
            break
        # all possible pairs
        if len(ids) < 2:
            continue
        total_pairs = len(ids) * (len(ids) - 1) // 2
        if total_pairs <= rem:
            for a, b in itertools.combinations(ids, 2):
                if a < b:
                    candidates.add((a, b))
            continue
        # sample pairs
        seen = set()
        while len(seen) < rem:
            a, b = random.sample(ids, 2)
            if a == b:
                continue
            pair = (a, b) if a < b else (b, a)
            seen.add(pair)
        candidates.update(seen)
    # fill random if needed
    all_ids = df[id_col].tolist()
    while len(candidates) < target:
        a, b = random.sample(all_ids, 2)
        if a == b:
            continue
        pair = (a, b) if a < b else (b, a)
        candidates.add(pair)
    # truncate
    cand = list(candidates)
    return cand[:target]


def compute_recall(candidates, truth_df):
    truth = set((min(r.lid, r.rid), max(r.lid, r.rid)) for _, r in truth_df.iterrows())
    cand_set = set(candidates)
    tp = len(truth & cand_set)
    return tp / len(truth) if truth else 0.0


def main():
    random.seed(42)
    # load data
    X1 = pd.read_csv("input/X1.csv")
    X2 = pd.read_csv("input/X2.csv")
    Y1 = pd.read_csv("input/Y1.csv")
    Y2 = pd.read_csv("input/Y2.csv")
    # generate candidates
    C1 = generate_candidates(X1, "id", "title", target=1000000, min_df=2, max_df=500)
    C2 = generate_candidates(X2, "id", "name", target=2000000, min_df=2, max_df=500)
    # evaluate recall
    r1 = compute_recall(C1, Y1)
    r2 = compute_recall(C2, Y2)
    print(f"Recall X1: {r1:.4f}, Recall X2: {r2:.4f}, Overall: {((r1*len(Y1)+r2*len(Y2))/(len(Y1)+len(Y2))):.4f}")
    # save submission
    sub = pd.DataFrame(C1 + C2, columns=["left_instance_id", "right_instance_id"])
    os.makedirs("working", exist_ok=True)
    sub.to_csv("working/submission.csv", index=False)


if __name__ == "__main__":
    main()
