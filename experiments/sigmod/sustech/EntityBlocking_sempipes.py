import re
from collections import defaultdict
from typing import *

import faiss
import pandas as pd
import torch
from CoSent.model import Model, encode, myTokenizer

nonsense = ["|", ",", "-", ":", "/", "+", "&"]

sandisk_patterns = [
    r"sandisk|cruzer|glide|256gb|usb",
    r"sandisk|cruzer|glide|32gb",
    r"sandisk|cruzer|extreme|16gb|usb",
    r"sandisk|extreme|class 10|128gb|sd",
    r"sandisk|n°282|x-series|64 16|sdhc",
    r"sandisk|ultra|performance|128gb|microsd|class 10",
    r"sandisk|extreme|pro|16gb|micro|sdhc",
    r"sandisk|accessoires montres / bracelets",
    r"sandisk|dual|style|otg|usb|go",
    r"sandisk|pami\?ci|sf-g|\(173473\)|microsdhc|300mb/s|95mb/s",
    r"sandisk|lsdmi8gbbbeu300a|lsdmi16gbb1eu300a",
    r"sandisk|cruzer carte|clase najwy\?szej|32gb|sdhc",
    r"sandisk|professional|\(niebieski\)|usb|\(cl\.4\)\+",
    r"sandisk|\(sdsqxaf-064g-gn6ma\)|extreme|64gb",
    r"sandisk|ultra|plus|32gb|sd",
    r"sandisk|cruzer|300mb/s|16gb|sdhc",
    r"sandisk|micro|v30|128gb|micro|\(bis de class",
    r"sandisk|pami\?ci|ultra|line|32gb|usb",
    r"sandisk|\(niebieski\)|\(4187407\)|sdhc",
    r"sandisk|pami\?\?|440/400mb/s|128gb|usb",
    r"sandisk|clé|glide|128gb",
    r"sandisk|\(DTSE9G2/128GB\)|LSDMI128BBEU633A",
    r"sandisk|10 Class|karta|jumpdrive",
    r"sandisk|sda10/128gb|\(mk483394661\)",
    r"sandisk|sdsquni-256g-gn6ma|lsd16gcrbeu1000|3502470",
    r"sandisk|ljdv10-32gabe|size 128",
    r"sandisk|go!/capturer|extreme",
    r"sandisk|fit 128gb",
    r"sandisk|memoria zu|cruzer|sd",
    r"sandisk|professional|80mo/s|uhs-i\n|16gb|32gb",
    r"tarjeta|sd|16gb|ush-i",
    r"sandisk|sdhc|android style|minneskort",
]


def block_x2(dataset: pd.DataFrame, max_limit):
    """
    Generate X2 result pairs using features from sem_extract_features

    :param dataset: feature dataframe of X2.csv after extracting features
    :param max_limit: the maximum number of output pairs
    :return: a list of output pairs
    """
    model_path = f"experiments/sigmod/sempipes_sustech/CoSent/x2_model/base_model_epoch_{38}.bin"
    my_model = Model(
        config_path="experiments/sigmod/sempipes_sustech/CoSent/prajjwal1_bert-tiny/config.json",
        bert_path="experiments/sigmod/sempipes_sustech/CoSent/prajjwal1_bert-tiny/pytorch_model.bin",
    )
    # load checkpoint safely: handle wrapped checkpoints and remove unwanted keys (e.g. position_ids)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # filter out keys that the model does not expect (position_ids cause mismatch in some checkpoints)
    if isinstance(state, dict):
        filtered_state = {k: v for k, v in state.items() if "position_ids" not in k}
    else:
        filtered_state = state
    res = my_model.load_state_dict(filtered_state, strict=False)
    try:
        print("Model load result - missing keys:", res.missing_keys, "unexpected keys:", res.unexpected_keys)
    except Exception:
        # older pytorch may return dict; print safely
        print("Model load result:", res)
    tokenizer = myTokenizer("experiments/sigmod/sempipes_sustech/CoSent/prajjwal1_bert-tiny")
    encodings = encode(model=my_model, sentences=dataset["name"], tokenizer=tokenizer)

    ids = dataset["id"].values
    series_list = dataset["series"].values
    pat_hb_list = dataset["pat_hb"].values
    brand_list = dataset["brand"].values
    capacity_list = dataset["capacity"].values
    mem_list = dataset["mem_type"].values
    product_list = dataset["type"].values
    model_list = dataset["model"].values
    name_list = dataset["normalized_name"].values if "normalized_name" in dataset.columns else dataset["name"].values
    item_code_list = dataset["item_code"].values
    hybrid_list = dataset["hybrid"].values
    long_num_list = dataset["long_num"].values

    buckets: Dict[str, List] = defaultdict(list)
    special_buckets: Dict[str, List] = defaultdict(list)
    confident_buckets: Dict[str, List] = defaultdict(list)

    # Additional blocking buckets for better recall (inspired by place5)
    item_code_buckets: Dict[str, List] = defaultdict(list)
    series_buckets: Dict[str, List] = defaultdict(list)
    brand_capacity_buckets: Dict[str, List] = defaultdict(list)
    brand_model_buckets: Dict[str, List] = defaultdict(list)

    for idx in range(dataset.shape[0]):
        buckets[brand_list[idx]].append(idx)

        # Item code blocking (very specific, high precision)
        if item_code_list[idx] != "0":
            item_code_buckets[item_code_list[idx]].append(idx)

        # Series blocking (brand-specific family)
        if series_list[idx] != "0":
            series_buckets[series_list[idx]].append(idx)
            # Brand + series combination
            if brand_list[idx] != "0":
                brand_series_key = f"{brand_list[idx]}_{series_list[idx]}"
                series_buckets[brand_series_key].append(idx)

        # Brand + capacity blocking (strong pattern)
        if brand_list[idx] != "0" and capacity_list[idx] != "0":
            brand_capacity_key = f"{brand_list[idx]}_{capacity_list[idx]}"
            brand_capacity_buckets[brand_capacity_key].append(idx)

        # Brand + model blocking
        if brand_list[idx] != "0" and model_list[idx] != "0":
            brand_model_key = f"{brand_list[idx]}_{model_list[idx]}"
            brand_model_buckets[brand_model_key].append(idx)

        if hybrid_list[idx] != "0" or long_num_list[idx] != "0":
            special_buckets[hybrid_list[idx] + long_num_list[idx]].append(idx)
        elif brand_list[idx] == "sandisk":
            for pattern in sandisk_patterns:
                name_copy = name_list[idx] + " " + capacity_list[idx]
                result_re = sorted(set(re.findall(pattern, name_copy)))
                if len(result_re) == len(pattern.split("|")):
                    confident_buckets["_".join(result_re)].append(idx)
                    break

    visited_set = set()
    candidate_pairs = []

    # Process item_code buckets (very specific, high precision)
    for key in item_code_buckets.keys():
        bucket = item_code_buckets[key]
        if len(bucket) > 100:  # Increased limit to generate more pairs
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, 0))

    # Process brand+capacity buckets (strong pattern)
    for key in brand_capacity_buckets.keys():
        bucket = brand_capacity_buckets[key]
        if len(bucket) > 300:  # Increased limit to generate more pairs
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, 0))

    # Process brand+model buckets
    for key in brand_model_buckets.keys():
        bucket = brand_model_buckets[key]
        if len(bucket) > 200:  # Increased limit to generate more pairs
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, 0))

    # Process series buckets
    for key in series_buckets.keys():
        bucket = series_buckets[key]
        if len(bucket) > 200:  # Increased limit to generate more pairs
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, 0))

    for key in confident_buckets.keys():
        bucket = confident_buckets[key]
        # Increased threshold to generate more pairs
        if len(bucket) > 1000:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, 0))

    for key in special_buckets.keys():
        bucket = special_buckets[key]
        # Increased threshold to generate more pairs
        if len(bucket) > 50:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, 0))

    for key in buckets.keys():
        bucket = buckets[key]
        embedding_matrix = encodings[bucket]
        # Maximum search for maximum recall - find as many candidates as possible
        if key == "sandisk":
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 16)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 1536
            D, I = index_model.search(embedding_matrix, min(400, len(bucket)))  # Increased to 400
        elif key == "sony":
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 12)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 1536
            D, I = index_model.search(embedding_matrix, min(400, len(bucket)))  # Increased to 400
        elif key == "0":
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 1536
            D, I = index_model.search(embedding_matrix, min(400, len(bucket)))  # Increased to 400
        elif key in ["lexar", "pny", "transcend"]:
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 1024
            D, I = index_model.search(embedding_matrix, min(300, len(bucket)))  # Increased to 300
        else:
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 1024
            D, I = index_model.search(embedding_matrix, min(300, len(bucket)))  # Increased to 300
        # DRASTIC CHANGE: Accept ALL FAISS pairs! Let FAISS distance be the only filter.
        # Only filter obvious negatives (same item, already visited, or clear mismatches)
        for i in range(len(D)):
            for j in range(len(D[0])):
                index1 = bucket[i]
                index2 = bucket[I[i][j]]
                s1 = ids[index1]
                s2 = ids[index2]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)

                # ACCEPT ALL PAIRS - NO FILTERING!
                # Any filtering causes recall loss. FAISS distance will handle ranking.
                candidate_pairs.append((small, large, D[i][j]))

    # Sort by FAISS distance (lower is better) and take top max_limit
    if len(candidate_pairs) > max_limit:
        candidate_pairs.sort(key=lambda x: x[2])  # Sort by distance (x[2] is the FAISS distance)
        candidate_pairs = candidate_pairs[:max_limit]

    candidate_pairs = [(x[0], x[1]) for x in candidate_pairs]

    return candidate_pairs
