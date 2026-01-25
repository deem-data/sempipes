from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name
from tqdm import tqdm

import sempipes
from sempipes.logging import get_logger
from sempipes.optimisers.colopro import optimise_colopro

if TYPE_CHECKING:
    from experiments.colopro import Setup

logger = get_logger()


class TestPipeline(ABC):
    OPERATOR_NAME = "op_to_evolve"
    TEST_SIZE = 0.25

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def score(self, y_true, y_pred) -> float:
        pass

    def requires_probabilities(self) -> bool:
        return False

    @abstractmethod
    def pipeline_with_all_data(self, seed):
        pass

    @abstractmethod
    def pipeline_with_train_data(self, seed):
        pass

    def optimize(self, seed, setup: Setup):
        sempipes.update_config(llm_for_code_generation=setup.llm_for_code_generation)

        pipeline_to_optimise, env_variables = self.pipeline_with_train_data(seed)
        outcomes = self._optimize_pipeline(pipeline_to_optimise, setup, env_variables)

        scores_over_time = []

        for max_trial in tqdm(range(0, setup.num_trials + 1), desc="Evaluating operator states"):
            outcomes_until_max_trial = [outcome for outcome in outcomes if outcome.search_node.trial <= max_trial]
            best_outcome = max(outcomes_until_max_trial, key=lambda x: (x.score, -x.search_node.trial))
            pipeline, env_variables = self.pipeline_with_all_data(seed)

            from_trial = best_outcome.search_node.trial
            try:
                score = self._evaluate(seed, pipeline, env_variables, operator_state=best_outcome.state)
            except Exception as e:
                logger.error(f"Error evaluating operator state: {e}", exc_info=True)
                score = None
            scores_over_time.append((max_trial, from_trial, best_outcome.score, score))

        return scores_over_time

    def _optimize_pipeline(self, pipeline, setup, additional_env_variables):
        search_policy = setup.search.clone_empty()
        outcomes = optimise_colopro(
            pipeline,
            TestPipeline.OPERATOR_NAME,
            num_trials=setup.num_trials,
            scoring=self.scoring,
            search=search_policy,
            cv=setup.cv,
            run_name=self.name,
            n_jobs_for_evaluation=-1,
            additional_env_variables=additional_env_variables,
        )

        return outcomes

    def _evaluate(self, seed, pipeline, additional_env_variables, operator_state=None):
        if operator_state is None:
            data_op = find_node_by_name(pipeline, self.OPERATOR_NAME)
            empty_state = data_op._skrub_impl.estimator.empty_state()
            operator_state = empty_state

        state_log = ("#" * 80) + "\n" + f"Operator state: {operator_state}" + "\n" + ("#" * 80)

        logger.info(state_log)

        np.random.seed(seed)

        data = additional_env_variables["data"]
        labels = additional_env_variables["labels"]

        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.TEST_SIZE, random_state=seed
        )

        train_env = pipeline.skb.get_data()
        test_env = pipeline.skb.get_data()

        for key, value in additional_env_variables.items():
            if key not in ["data", "labels"]:
                train_env[key] = value
                test_env[key] = value

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

        train_env["data"] = train_data
        train_env["labels"] = train_labels
        train_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = operator_state
        learner.fit(train_env)

        test_env["data"] = test_data
        test_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = operator_state
        if self.requires_probabilities():
            y_pred = learner.predict_proba(test_env)
        else:
            y_pred = learner.predict(test_env)
        score = self.score(test_labels, y_pred)

        score_log = ("%" * 80) + "\n" + f"Score: {score}" + "\n" + ("%" * 80)

        logger.info(score_log)

        return score
