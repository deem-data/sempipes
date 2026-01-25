import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def block_with_attr(X, attr):
    """
    Perform blocking using attr.
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

    # remove duplicate pairs and take union
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


def save_output(
    X1_candidate_pairs, X2_candidate_pairs
):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("data/output.csv", index=False)


# read the datasets
X1 = pd.read_csv("../../../../Downloads/secret/Z1.csv")
X2 = pd.read_csv("../../../../Downloads/secret/Z2.csv")

# perform blocking
X1_candidate_pairs = block_with_attr(X1, attr="title")
X2_candidate_pairs = block_with_attr(X2, attr="name")

# save results
save_output(X1_candidate_pairs, X2_candidate_pairs)
