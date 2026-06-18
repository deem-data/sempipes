import sys
from pathlib import Path

import pandas as pd

from experiments.sigmod.rutgers.execute_sempipes_medium_optimized import (
    BlockingModel_X2,
    _create_env,
    _pipeline,
)
from sempipes.optimisers.search_policy import _ROOT_TRIAL
from sempipes.optimisers.trajectory import load_trajectory_from_json


def run_from_trajectory(trajectory_path: str, mode: int = 0, skip_root: bool = True):
    path = Path(trajectory_path)
    trajectory = load_trajectory_from_json(path)

    candidates = trajectory.outcomes
    if skip_root:
        non_root = [o for o in trajectory.outcomes if o.search_node.trial != _ROOT_TRIAL]
        if non_root:
            candidates = non_root
    best = max(candidates, key=lambda o: (o.score, -o.search_node.trial))
    print(f"Best outcome: trial={best.search_node.trial}, score={best.score}")
    print(f"Best outcome states: {best.states}")

    operator_name = "discover_additional_blocking_features"
    operator_name2 = "extract_x2_features_fixed"

    gen_state = best.states.get(operator_name) if best.states.exists_for(operator_name) else None
    extract_state = best.states.get(operator_name2) if best.states.exists_for(operator_name2) else None

    if mode == 0:
        X2 = pd.read_csv("experiments/sigmod/data/X2.csv")
        train_X = X2.copy()
        train_labels = pd.read_csv("experiments/sigmod/data/Y2.csv")
        test_data = pd.read_csv("experiments/sigmod/hidden_data/Z2.csv")
        test_labels = pd.read_csv("experiments/sigmod/hidden_data/Y2.csv")
    else:
        X2 = pd.read_csv("experiments/sigmod/data/X2.csv")
        train_X = X2.copy()
        train_labels = pd.read_csv("experiments/sigmod/data/Y2.csv")
        test_data = X2.copy()
        test_labels = train_labels.copy()

    train_X["name"] = train_X["name"].str.lower()
    test_data["name"] = test_data["name"].str.lower()

    pipeline = _pipeline(operator_name, operator_name2)

    def make_env(X, labels, gen_s, ext_s):
        env = _create_env(X, labels, operator_name, operator_name2, gen_s)
        env[f"sempipes_prefitted_state__{operator_name2}"] = ext_s
        return env

    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(make_env(train_X, train_labels, gen_state, extract_state))
    results = learner.predict(make_env(test_data, test_labels, gen_state, extract_state))

    predicted_df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results, columns=["left_instance_id", "right_instance_id"])
    predicted_df["left_right"] = predicted_df["left_instance_id"].astype(str) + predicted_df["right_instance_id"].astype(str)
    ground_truth = test_labels.copy()
    ground_truth["left_right"] = ground_truth["lid"].astype(str) + ground_truth["rid"].astype(str)
    inter = set(predicted_df["left_right"]) & set(ground_truth["left_right"])
    recall = round(len(inter) / len(ground_truth) if len(ground_truth) > 0 else 0.0, 3)
    print(f"X2 recall: {recall}")


if __name__ == "__main__":
    # traj_path = ".sempipes_trajectories/sigmod_rutgers_sempipes_3_optimized_20260329_044712_273de3e8.json" 
    #traj_path = ".sempipes_trajectories/sigmod_rutgers_sempipes_3_optimized_ubc_20260329_052130_6fc56894.json" #0.255
    traj_path = ".sempipes_trajectories/sigmod_rutgers_sempipes_3_optimized_20260613_003910_ca15735f.json"
    mode = 0
    skip_root = True 
    # skip_root = False
    run_from_trajectory(traj_path, mode, skip_root)
