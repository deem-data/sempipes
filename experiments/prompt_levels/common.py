from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split
from skrub._data_ops._evaluation import find_node_by_name
from tqdm import tqdm

import sempipes
from experiments.colopro import TestPipeline
from sempipes.logging import get_logger
from sempipes.optimisers.trajectory import load_trajectory_from_json

if TYPE_CHECKING:
    from experiments.colopro import Setup

logger = get_logger()


class PromptLevelTestPipeline(TestPipeline):
    """TestPipeline extensions used by prompt-level experiments only."""

    def _operator_state_from_outcome(self, outcome):
        if outcome.states.exists_for(self.OPERATOR_NAME):
            return outcome.states.get(self.OPERATOR_NAME)
        legacy_state = getattr(outcome, "state", None)
        if legacy_state is not None:
            return legacy_state
        raise KeyError(f"No operator state found for {self.OPERATOR_NAME!r} in outcome")

    def _evaluate_with_outcome(self, seed, pipeline, additional_env_variables, outcome):
        operator_state = self._operator_state_from_outcome(outcome)
        return self._evaluate(seed, pipeline, additional_env_variables, operator_state=operator_state)

    def _evaluate_outcomes_over_time(self, seed, outcomes, num_trials: int):
        scores_over_time = []
        for max_trial in tqdm(range(0, num_trials + 1), desc="Evaluating operator states"):
            outcomes_until_max_trial = [o for o in outcomes if o.search_node.trial <= max_trial]
            best_outcome = max(outcomes_until_max_trial, key=lambda x: (x.score, -x.search_node.trial))
            pipeline, env_variables = self.pipeline_with_all_data(seed)
            from_trial = best_outcome.search_node.trial
            try:
                score = self._evaluate_with_outcome(seed, pipeline, env_variables, best_outcome)
            except Exception as e:
                logger.error(f"Error evaluating operator state: {e}", exc_info=True)
                score = None
            scores_over_time.append((max_trial, from_trial, best_outcome.score, score))
        return scores_over_time

    def optimize(self, seed, setup: Setup):
        sempipes.update_config(llm_for_code_generation=setup.llm_for_code_generation)
        pipeline_to_optimise, env_variables = self.pipeline_with_train_data(seed)
        outcomes = self._optimize_pipeline(pipeline_to_optimise, setup, env_variables)
        return self._evaluate_outcomes_over_time(seed, outcomes, setup.num_trials)

    def evaluate_trajectory_scores_over_time(
        self, seed, trajectory_path: Path | str, num_trials: int | None = None
    ):
        trajectory = load_trajectory_from_json(trajectory_path)
        outcomes = trajectory.outcomes
        if num_trials is None:
            num_trials = max(o.search_node.trial for o in outcomes)
        return self._evaluate_outcomes_over_time(seed, outcomes, num_trials)

    def _evaluate_with_prompt_generation(self, seed, pipeline, additional_env_variables):
        np.random.seed(seed)

        data = additional_env_variables["data"]
        labels = additional_env_variables["labels"]

        train_data, _, train_labels, _ = train_test_split(
            data, labels, test_size=self.TEST_SIZE, random_state=seed
        )

        train_env = pipeline.skb.get_data()
        for key, value in additional_env_variables.items():
            train_env[key] = value
        train_env["data"] = train_data
        train_env["labels"] = train_labels
        train_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = None

        pipeline_clone = pipeline.skb.clone()
        operator_op = find_node_by_name(pipeline_clone, self.OPERATOR_NAME)
        operator_op.skb.eval(train_env)
        operator_state = operator_op._skrub_impl.estimator_.state_after_fit()

        return self._evaluate(seed, pipeline, additional_env_variables, operator_state=operator_state)
