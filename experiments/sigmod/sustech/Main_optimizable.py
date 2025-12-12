import pandas as pd
import skrub
from EntityBlocking import block_x2
from FeatureExtracting_sempipes_optimizable import extract_x2_sempipes
from sklearn.base import BaseEstimator

import sempipes
from sempipes.optimisers import EvolutionarySearch, optimise_colopro


class BlockingModel(BaseEstimator):
    def __init__(self, size_of_output=2000000):
        self.size_of_output = size_of_output

    def fit(self, X, y):
        return self

    def predict(self, X):
        return block_x2(X, self.size_of_output)


def calculate_recall(estimator, X, y):
    """
    Sklearn-compatible scorer function that calculates recall.

    Args:
        estimator: The fitted estimator (pipeline)
        X: Test features (DataFrame or dict with '_skrub_X' key)
        y: Test labels (DataFrame with 'lid' and 'rid' columns from skrub's cross_validate)

    Returns:
        float: Recall score
    """
    predictions = estimator.predict(X)
    if isinstance(predictions, list):
        predicted_df = pd.DataFrame(predictions, columns=["left_instance_id", "right_instance_id"])
    elif isinstance(predictions, pd.DataFrame):
        predicted_df = predictions.copy()
    else:
        predicted_df = pd.DataFrame(predictions)
        if "left_instance_id" not in predicted_df.columns or "right_instance_id" not in predicted_df.columns:
            if len(predicted_df.columns) == 2:
                predicted_df.columns = ["left_instance_id", "right_instance_id"]
            else:
                raise ValueError(f"Unexpected prediction format: {type(predictions)}")

    # Ensure ground_truth is a DataFrame
    if not isinstance(y, pd.DataFrame):
        ground_truth = pd.DataFrame(y)
    else:
        ground_truth = y.copy()

    # Verify it has the expected columns
    if "lid" not in ground_truth.columns or "rid" not in ground_truth.columns:
        raise ValueError(f"Ground truth labels must have 'lid' and 'rid' columns. Got: {ground_truth.columns.tolist()}")

    # Calculate recall
    predicted_df["left_right"] = predicted_df["left_instance_id"].astype(str) + predicted_df[
        "right_instance_id"
    ].astype(str)
    predicted_values = predicted_df["left_right"].values

    ground_truth["left_right"] = ground_truth["lid"].astype(str) + ground_truth["rid"].astype(str)
    reference_values = ground_truth["left_right"].values

    inter = set.intersection(set(predicted_values), set(reference_values))
    recall = len(inter) / len(reference_values) if len(reference_values) > 0 else 0.0

    return round(recall, 3)


def _pipeline(operator_name):
    data_ref = skrub.var("data_original_x2").skb.mark_as_X()
    y = skrub.var("dummy_y").skb.mark_as_y()
    features = extract_x2_sempipes(data_ref, operator_name)
    return features.skb.apply(BlockingModel(2000000), y=y)


def _create_env(X, y, operator_name, state):
    """Create environment dictionary for learner."""
    dummy_y = pd.Series([0] * len(X), name="dummy")

    return {
        "_skrub_X": X,
        "_skrub_y": dummy_y,
        "data_original_x2": X,
        "dummy_y": dummy_y,
        f"sempipes_memory__{operator_name}": None,
        f"sempipes_pipeline_summary__{operator_name}": None,
        f"sempipes_prefitted_state__{operator_name}": state,
        f"sempipes_inspirations__{operator_name}": None,
    }


def main_sempipes_optimizable_X2(data_path_small2, data_path_hidden2, mode, base_path_small, base_path_hidden):
    # Configure sempipes for X2 optimization.
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
        llm_for_batch_processing=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
    )

    sample = pd.read_csv(data_path_small2)
    sample_labels = pd.read_csv(base_path_small + "/Y2.csv")

    train_X = sample
    train_labels = sample_labels

    if mode == 0:
        test_data = pd.read_csv(data_path_hidden2)
        test_labels = pd.read_csv(base_path_hidden + "/Y2.csv")
    else:
        test_data = pd.read_csv(data_path_small2)
        test_labels = pd.read_csv(base_path_small + "/Y2.csv")

    sample["name"] = sample["name"].str.lower()
    test_data["name"] = test_data["name"].str.lower()

    # W/ optimization
    operator_name = "extract_x2_features"
    pipeline_to_optimise = _pipeline(operator_name)

    def recall_scorer_with_labels(estimator, X_test, y=None, **kwargs):
        if isinstance(X_test, dict):
            X_test_data = X_test.get("_skrub_X", X_test)
        else:
            X_test_data = X_test
        if "id" in X_test_data.columns:
            test_ids = set(X_test_data["id"].values)
            test_labels = train_labels[train_labels["lid"].isin(test_ids) & train_labels["rid"].isin(test_ids)].copy()
            return calculate_recall(estimator, X_test, y=test_labels, **kwargs)
        else:
            return calculate_recall(estimator, X_test, y=train_labels, **kwargs)

    outcomes = optimise_colopro(
        pipeline_to_optimise,
        operator_name,
        scoring=recall_scorer_with_labels,
        cv=5,
        num_trials=24,
        search=EvolutionarySearch(population_size=6),
        additional_env_variables={"data_original_x2": train_X, "dummy_y": pd.Series([0] * len(train_X), name="dummy")},
    )

    best_outcome = max(outcomes, key=lambda x: (x.score, -x.search_node.trial))
    print(f"Best outcome score after optimization on train CV: {best_outcome.score}, state: {best_outcome.state}")

    learner_optimized = pipeline_to_optimise.skb.make_learner(fitted=False, keep_subsampling=False)
    learner_optimized.fit(_create_env(train_X, None, operator_name, best_outcome.state))
    optimized_results = learner_optimized.predict(_create_env(test_data, None, operator_name, best_outcome.state))

    return optimized_results
