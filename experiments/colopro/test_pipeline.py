from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name
from tqdm import tqdm

import sempipes
from sempipes.optimisers.colopro import optimise_colopro

if TYPE_CHECKING:
    from experiments.colopro import Setup


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
    def pipeline_with_all_data(self, seed) -> DataOp:
        pass

    @abstractmethod
    def pipeline_with_train_data(self, seed) -> DataOp:
        pass

    def optimize(self, seed, setup: Setup):
        sempipes.update_config(llm_for_code_generation=setup.llm_for_code_generation)

        pipeline_to_optimise = self.pipeline_with_train_data(seed)
        outcomes = self._optimize_pipeline(pipeline_to_optimise, setup)

        scores_over_time = []

        for max_trial in tqdm(range(0, setup.num_trials + 1, 2), desc="Evaluating operator states"):
            outcomes_until_max_trial = [outcome for outcome in outcomes if outcome.search_node.trial <= max_trial]
            best_outcome = max(outcomes_until_max_trial, key=lambda x: (x.score, -x.search_node.trial))
            pipeline = self.pipeline_with_all_data(seed)

            from_trial = best_outcome.search_node.trial
            try:
                score = self._evaluate(seed, pipeline, operator_state=best_outcome.state)
            except Exception as e:
                print("Error evaluating operator state:", e)
                import traceback

                traceback.print_exc()
                score = None
            scores_over_time.append((max_trial, from_trial, best_outcome.score, score))

        return scores_over_time

    def _optimize_pipeline(self, pipeline, setup):
        search_policy = setup.search.clone_empty()
        outcomes = optimise_colopro(
            pipeline,
            TestPipeline.OPERATOR_NAME,
            num_trials=setup.num_trials,
            scoring=self.scoring,
            search=search_policy,
            cv=10,
            run_name=self.name,
        )

        return outcomes

    def _evaluate(self, seed, pipeline, operator_state=None):
        if operator_state is None:
            data_op = find_node_by_name(pipeline, self.OPERATOR_NAME)
            empty_state = data_op._skrub_impl.estimator.empty_state()
            operator_state = empty_state

        print("#" * 80)
        print("Operator state:", operator_state)
        print("#" * 80)

        np.random.seed(seed)
        split = pipeline.skb.train_test_split(test_size=self.TEST_SIZE, random_state=seed, stratify=self.stratify_by)
        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

        train_env = split["train"]
        train_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = operator_state
        learner.fit(train_env)

        test_env = split["test"]
        test_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = operator_state
        if self.requires_probabilities():
            y_pred = learner.predict_proba(test_env)
        else:
            y_pred = learner.predict(test_env)
        score = self.score(split["test"]["_skrub_y"], y_pred)
        print("%" * 80)
        print(score)
        print("%" * 80)

        return score
