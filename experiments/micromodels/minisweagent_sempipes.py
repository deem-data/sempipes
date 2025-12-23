import json
import sempipes
import skrub
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def sempipes_pipeline():
    data = skrub.var("responses")
    y = data["emotional_reaction_level"].skb.mark_as_y()
    data = data[["response_post"]].skb.mark_as_X()

    data = data.sem_extract_features(
        nl_prompt="""
        Extract the emotional reaction level from the response post.
        """,
    )

scores = []
for split_index, seed in enumerate([42, 1337, 2025, 7321, 98765]):
    print(f"Processing split {split_index}")
    np.random.seed(seed)

    # Load data
    with open("experiments/micromodels/empathy.json") as f:
        data = json.load(f)

    X = []
    y = []

    for entry in data.values():
        response = entry["response_post"]
        emo_feat = extract_emotional_level(response)
        interp_feat = extract_interpretation_level(response)
        expl_feat = extract_exploration_level(response)
        X.append([emo_feat, interp_feat, expl_feat])
        y.append(int(entry["emotional_reactions"]["level"]))  # Target is emotional_reactions level

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    # Train ExplainableBoostingClassifier
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = ebm.predict(X_test)
    score = f1_score(y_test, y_pred, average="micro")
    print(f"F1 score on {split_index}: {score}")
    scores.append(score)

print("\nMean final score: ", np.mean(scores), np.std(scores))
