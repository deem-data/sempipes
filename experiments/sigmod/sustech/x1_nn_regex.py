import re
from collections import defaultdict
from typing import *

import faiss
import pandas as pd
from clean import clean
from sentence_transformers import SentenceTransformer


def x1_test(data: pd.DataFrame, limit: int, model_path: str) -> list:
    """
    Generate X1 result pairs

    :param data: raw data read from X1.csv
    :param limit: the maximum number of output pairs
    :param model_path: the path of SentenceTransformer model
    :return: a list of output pairs
    """
    features = clean(data)
    return x_test_blocking_x1(data, features, limit, model_path)


def x_test_blocking_x1(data: pd.DataFrame, features: pd.DataFrame, limit: int, model_path: str) -> list:
    model = SentenceTransformer(model_path, device="cpu")
    encodings = model.encode(sentences=data["title"], batch_size=256, normalize_embeddings=True)
    topk = 50
    candidate_pairs: List[Tuple[int, int, float]] = []
    ram_capacity_list = features["ram_capacity"].values
    cpu_model_list = features["cpu_model"].values
    title_list = (
        features["normalized_title"].values if "normalized_title" in features.columns else features["title"].values
    )
    family_list = features["family"].values
    identification_list = defaultdict(list)
    reg_list = defaultdict(list)
    number_list = defaultdict(list)
    regex_pattern = re.compile("(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[^\W_gGM]{6,}")  # FIXME Maybe sempipes
    number_pattern = re.compile("[0-9]{4,}")  # FIXME Maybe sempipes
    buckets = defaultdict(list)
    for idx in range(data.shape[0]):
        title = " ".join(sorted(set(title_list[idx].split())))

        regs = regex_pattern.findall(title)
        identification = " ".join(sorted(regs))
        reg_list[identification].append(idx)

        identification_list[title].append(idx)

        number_id = number_pattern.findall(title)
        number_id = " ".join(sorted(number_id))
        number_list[number_id].append(idx)

        brands = features["brand"][idx]
        for brand in brands:
            buckets[brand].append(idx)
        if len(brands) == 0:
            buckets["0"].append(idx)
    visited_set = set()
    ids = data["id"].values
    regex_pairs = []

    for key in identification_list:
        cluster = identification_list[key]
        if len(cluster) > 1:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    for key in reg_list:
        cluster = reg_list[key]
        if len(cluster) <= 5:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    for key in number_list:
        cluster = number_list[key]
        if len(cluster) <= 5:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    limit = limit - len(regex_pairs)
    if limit < 0:
        limit = 0

    for key in buckets:
        cluster = buckets[key]
        embedding_matrix = encodings[cluster]
        k = min(topk, len(cluster))
        index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        index_model.hnsw.efConstruction = 100
        index_model.add(embedding_matrix)
        index_model.hnsw.efSearch = 256
        D, I = index_model.search(embedding_matrix, k)
        for i in range(len(D)):
            for j in range(len(D[0])):
                index1 = cluster[i]
                index2 = cluster[I[i][j]]
                s1 = ids[index1]
                s2 = ids[index2]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = str(small) + " " + str(large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                if not (
                    ram_capacity_list[index1] == "0"
                    or ram_capacity_list[index2] == "0"
                    or ram_capacity_list[index1] == ram_capacity_list[index2]
                ):
                    if family_list[index1] != "x220" and family_list[index2] != "x220":
                        continue
                intersect = cpu_model_list[index1].intersection(cpu_model_list[index2])
                if not (len(cpu_model_list[index1]) == 0 or len(cpu_model_list[index2]) == 0 or len(intersect) != 0):
                    continue
                candidate_pairs.append((small, large, D[i][j]))

    candidate_pairs.sort(key=lambda x: x[2])
    candidate_pairs = candidate_pairs[:limit]
    output = list(map(lambda x: (x[0], x[1]), candidate_pairs))
    output.extend(regex_pairs)
    return output
