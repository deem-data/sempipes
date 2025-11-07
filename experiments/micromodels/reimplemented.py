# https://github.com/MichiganNLP/micromodels/tree/master/empathy
import random
import warnings

import numpy as np
import pandas as pd
import torch
from interpret.glassbox import ExplainableBoostingClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_DEVICE = "cuda:3"


def extract_rationales(sample_data, empathy_type):
    rationales = set({})
    non_zero_level_data = sample_data[sample_data[f"{empathy_type}_level"] > 0]
    serialized_rationales = non_zero_level_data[f"{empathy_type}_level_rationale"].dropna().unique()

    for serialized_rationale in serialized_rationales:
        for rationale in serialized_rationale.split("|"):
            if rationale != "":
                rationales.add(rationale)

    return rationales


def bert_micromodel(responses, rationales):
    model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1", device=_DEVICE)
    emb1 = model.encode(list(responses), normalize_embeddings=True)
    emb2 = model.encode(list(rationales), normalize_embeddings=True)
    sim = model.similarity(emb1, emb2)
    max_scores, _ = torch.max(sim, dim=1)
    return max_scores


all_data = pd.read_csv("experiments/micromodels/empathy.csv")

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    print(f"Processing split {split_index}")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_data, test_data = train_test_split(all_data, test_size=0.5, random_state=seed)

    print("\tExtracting rationales")
    emo_rationales = extract_rationales(train_data, "emotional_reaction")
    interpretation_rationales = extract_rationales(train_data, "interpretation")
    explorations_rationales = extract_rationales(train_data, "explorations")

    print("\tComputing train features via micromodels")
    feature_names = ["emotional_reactions", "interpretations", "explorations"]

    emo_scores = bert_micromodel(train_data.response, emo_rationales)
    interpretation_scores = bert_micromodel(train_data.response, interpretation_rationales)
    explorations_scores = bert_micromodel(train_data.response, explorations_rationales)

    train_features = torch.column_stack((emo_scores, interpretation_scores, explorations_scores)).tolist()

    print("\tTraining EBM classifier")
    emo_ebm = ExplainableBoostingClassifier(feature_names=feature_names)
    emo_ebm.fit(train_features, train_data.emotional_reaction_level)

    print("\tComputing test features via micromodels")
    test_emo_scores = bert_micromodel(test_data.response, emo_rationales)
    test_interpretation_scores = bert_micromodel(test_data.response, interpretation_rationales)
    test_explorations_scores = bert_micromodel(test_data.response, explorations_rationales)

    test_features = torch.column_stack((test_emo_scores, test_interpretation_scores, test_explorations_scores)).tolist()

    test_predictions = emo_ebm.predict(test_features)
    score = f1_score(test_data.emotional_reaction_level, test_predictions, average="micro")
    print(f"F1 score on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
