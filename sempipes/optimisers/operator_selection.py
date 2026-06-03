import math
from abc import ABC, abstractmethod
from graphlib import TopologicalSorter

from mabwiser.mab import MAB, LearningPolicy
from skrub._data_ops._data_ops import Apply
from skrub._data_ops._evaluation import DataOp, find_node_by_name, graph

from sempipes.logging import get_logger
from sempipes.operators.sem_extract_features.with_code import CodeBasedFeatureExtractor
from sempipes.operators.sem_gen_features_caafe import LLMFeatureGenerator
from sempipes.optimisers.search_policy import SearchPolicy

logger = get_logger()

_OPERATOR_NAME_PREFIX = "sempipes_prefitted_state__"


def _extract_operator_names(pipeline):
    env_for_inspection = pipeline.skb.get_data()
    all_operator_names = [
        name.split("__")[1] for name in env_for_inspection.keys() if name.startswith(_OPERATOR_NAME_PREFIX)
    ]
    all_operator_names = sorted(list(set(all_operator_names)))
    return all_operator_names


def _hop(current_node_id, dag, candidate_operator_names, dependent_operator_names):
    if current_node_id in dag["parents"]:
        children_ids = dag["parents"][current_node_id]
        for child_id in children_ids:
            child_node = dag["nodes"][child_id]
            if isinstance(child_node, DataOp) and child_node._skrub_impl.name in candidate_operator_names:
                dependent_operator_names.append(child_node._skrub_impl.name)
            _hop(child_id, dag, candidate_operator_names, dependent_operator_names)


def _collect_dependent_operators(pipeline: DataOp) -> dict[str, list[str]]:
    dag = graph(pipeline)
    all_operator_names = _extract_operator_names(pipeline)

    operator_dependencies = {}
    for op_of_interest_name in all_operator_names:
        op_id = next(
            (
                node_id
                for node_id, node in dag["nodes"].items()
                if isinstance(node, DataOp) and node._skrub_impl.name == op_of_interest_name
            ),
            None,
        )
        assert op_id is not None
        candidate_operator_names = [name for name in all_operator_names if name != op_of_interest_name]
        dependent_operator_names: list[str] = []
        _hop(op_id, dag, candidate_operator_names, dependent_operator_names)
        unique_dependent_operator_names = list(dict.fromkeys(dependent_operator_names))
        operator_dependencies[op_of_interest_name] = unique_dependent_operator_names

    return operator_dependencies


def _potentially_breaks_consumers(operator_name: str, pipeline: DataOp) -> bool:
    changes_output_columns = False

    op = find_node_by_name(pipeline, operator_name)
    assert isinstance(op._skrub_impl, Apply)

    if isinstance(op._skrub_impl.estimator, LLMFeatureGenerator):
        changes_output_columns = True

    if isinstance(op._skrub_impl.estimator, CodeBasedFeatureExtractor):
        if op._skrub_impl.estimator.output_columns is None or len(op._skrub_impl.estimator.output_columns) == 0:
            changes_output_columns = True

    return changes_output_columns


class OperatorSelectionPolicy(ABC):
    def __init__(self, dag_sink: DataOp, search: SearchPolicy):
        self.dag_sink = dag_sink
        self.search = search
        self.operator_dependencies = _collect_dependent_operators(dag_sink)

        self.all_operator_names = list(reversed(list(TopologicalSorter(self.operator_dependencies).static_order())))

        logger.info(f"OP-SELECTION> Operators in topological order: {self.all_operator_names}")

    @abstractmethod
    def select_operators_to_evolve(self, trial: int) -> list[str]:
        pass

    def as_dict(self) -> dict:
        return {"operator_dependencies": self.operator_dependencies, "all_operator_names": self.all_operator_names}


class UCBOperatorSelectionPolicy(OperatorSelectionPolicy):
    def __init__(self, dag_sink: DataOp, search: SearchPolicy):
        super().__init__(dag_sink, search)

        arms = self.all_operator_names
        self.bandit = MAB(arms, LearningPolicy.UCB1(alpha=0.2)) # alpha=0.5

    def _reward(self, trial: int) -> tuple[str, float]:
        assert trial > 0
        outcome_to_score = self.search.outcomes[trial]
        assert outcome_to_score is not None
        parent_outcome = next(
            (o for o in self.search.outcomes if o.search_node.trial == outcome_to_score.search_node.parent_trial), None
        )
        assert parent_outcome is not None
        improvement = outcome_to_score.score - parent_outcome.score
        assert outcome_to_score.search_node.operator_to_evolve is not None
        return outcome_to_score.search_node.operator_to_evolve, improvement

    def select_operators_to_evolve(self, trial: int) -> list[str]:
        assert trial > 0
        if trial < len(self.all_operator_names) + 1:
            chosen_operator = self.all_operator_names[trial - 1]
        elif trial == len(self.all_operator_names) + 1:
            logger.info("Done collecting initial rewards, switching to bandit-based operator selection")
            initial_rewards = []
            for initial_trial in range(1, trial):
                _, improvement = self._reward(initial_trial)
                initial_rewards.append(improvement)
            self.bandit.fit(self.all_operator_names, initial_rewards)
            chosen_operator = self.bandit.predict()
        else:
            previous_operator, improvement = self._reward(trial - 1)
            self.bandit.partial_fit([previous_operator], [improvement])
            chosen_operator = self.bandit.predict()

        if _potentially_breaks_consumers(chosen_operator, self.dag_sink):
            operators_to_evolve = [chosen_operator] + self.operator_dependencies[chosen_operator]
            logger.info(
                f"OP-SELECTION> Selected {chosen_operator} with dependents: {self.operator_dependencies[chosen_operator]}"
            )
        else:
            operators_to_evolve = [chosen_operator]
            logger.info(f"OP-SELECTION> Selected {chosen_operator} without dependents")

        return operators_to_evolve


class AdaEvolveGlobalNormOperatorSelectionPolicy(OperatorSelectionPolicy):
    """AdaEvolve Level-2 selector.

    Eq. 4: global-normalized reward r_t^(k)
    Eq. 5: decayed reward/visit updates R_t^(k), V_t^(k)
    Eq. 6: UCB selection over decayed mean reward + exploration bonus
    """

    def __init__(
        self,
        dag_sink: DataOp,
        search: SearchPolicy,
        rho: float = 0.7,
        ucb_c: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(dag_sink, search)
        self.rho = rho
        self.ucb_c = ucb_c
        self.eps = eps
        self.decayed_rewards = {operator: 0.0 for operator in self.all_operator_names}
        self.decayed_visits = {operator: 0.0 for operator in self.all_operator_names}
        self.raw_visits = {operator: 0 for operator in self.all_operator_names}

    def _find_outcome(self, trial: int):
        return next((outcome for outcome in self.search.outcomes if outcome.search_node.trial == trial), None)

    def _update_stats_from_previous_trial(self, trial: int):
        if trial <= 1:
            return

        previous_outcome = self._find_outcome(trial - 1)
        if previous_outcome is None or previous_outcome.search_node.operator_to_evolve is None:
            return

        operator = previous_outcome.search_node.operator_to_evolve
        parent_score = previous_outcome.search_node.parent_score
        if parent_score is None:
            reward = 0.0
        else:
            # Eq. 4-style global normalization with stability guards.
            improvement = previous_outcome.score - parent_score
            best_score_so_far = max(
                outcome.score for outcome in self.search.outcomes if outcome.search_node.trial <= previous_outcome.search_node.trial
            )
            reward = improvement / max(abs(best_score_so_far), self.eps)

        # Eq. 5-style decayed reward and decayed visit updates.
        self.decayed_rewards[operator] = self.rho * self.decayed_rewards[operator] + reward
        self.decayed_visits[operator] = self.rho * self.decayed_visits[operator] + 1.0
        self.raw_visits[operator] += 1

    def _select_operator(self) -> str:
        unvisited = [operator for operator in self.all_operator_names if self.raw_visits[operator] == 0]
        if unvisited:
            return unvisited[0]

        total_visits = sum(self.raw_visits.values())
        total_visits_for_log = max(total_visits, 1)
        best_operator = self.all_operator_names[0]
        best_ucb_value = float("-inf")
        for operator in self.all_operator_names:
            exploitation = self.decayed_rewards[operator] / max(self.decayed_visits[operator], self.eps)
            exploration = self.ucb_c * math.sqrt(math.log(total_visits_for_log) / max(self.raw_visits[operator], 1))
            ucb_value = exploitation + exploration
            logger.debug(
                "OP-SELECTION> AdaEvolve candidate=%s exploitation=%.6f exploration=%.6f ucb=%.6f visits=%d",
                operator,
                exploitation,
                exploration,
                ucb_value,
                self.raw_visits[operator],
            )
            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_operator = operator

        return best_operator

    def select_operators_to_evolve(self, trial: int) -> list[str]:
        assert trial > 0
        self._update_stats_from_previous_trial(trial)
        chosen_operator = self._select_operator()
        logger.debug(
            "OP-SELECTION> AdaEvolve selected=%s trial=%d decayed_reward=%.6f decayed_visits=%.6f raw_visits=%d",
            chosen_operator,
            trial,
            self.decayed_rewards[chosen_operator],
            self.decayed_visits[chosen_operator],
            self.raw_visits[chosen_operator],
        )

        if _potentially_breaks_consumers(chosen_operator, self.dag_sink):
            operators_to_evolve = [chosen_operator] + self.operator_dependencies[chosen_operator]
            logger.info(
                f"OP-SELECTION> AdaEvolve global-norm selected {chosen_operator} with dependents: "
                f"{self.operator_dependencies[chosen_operator]}"
            )
        else:
            operators_to_evolve = [chosen_operator]
            logger.info(f"OP-SELECTION> AdaEvolve global-norm selected {chosen_operator} without dependents")

        return operators_to_evolve

    def as_dict(self) -> dict:
        base = super().as_dict()
        base.update(
            {
                "policy": "adaevolve_global_norm",
                "rho": self.rho,
                "ucb_c": self.ucb_c,
                "eps": self.eps,
                "decayed_rewards": self.decayed_rewards,
                "decayed_visits": self.decayed_visits,
                "raw_visits": self.raw_visits,
            }
        )
        return base


# Backward-compatible alias for earlier in-flight naming.
UCBGlobalNormOperatorSelectionPolicy = AdaEvolveGlobalNormOperatorSelectionPolicy


class RoundRobinOperatorSelectionPolicy(OperatorSelectionPolicy):
    def select_operators_to_evolve(self, trial: int) -> list[str]:
        assert trial > 0
        num_operators = len(self.all_operator_names)
        idx = (trial - 1) % num_operators
        chosen_operator = self.all_operator_names[idx]
        logger.info(f"OP-SELECTION> Round-robin: selected {chosen_operator} (trial {trial}, index {idx})")

        if _potentially_breaks_consumers(chosen_operator, self.dag_sink):
            operators_to_evolve = [chosen_operator] + self.operator_dependencies[chosen_operator]
            logger.info(
                f"OP-SELECTION> Selected {chosen_operator} with dependents: {self.operator_dependencies[chosen_operator]}"
            )
        else:
            operators_to_evolve = [chosen_operator]
            logger.info(f"OP-SELECTION> Selected {chosen_operator} without dependents")

        return operators_to_evolve

class AllOperatorsPolicy(OperatorSelectionPolicy):
    def select_operators_to_evolve(self, trial: int) -> list[str]:
        assert trial > 0
        logger.info(f"OP-SELECTION> All-operators: evolving all {len(self.all_operator_names)} operators")
        return list(self.all_operator_names)


class FixedOpPolicy(OperatorSelectionPolicy):
    def __init__(self, dag_sink: DataOp, search: SearchPolicy, fixed_operator_name: str):
        super().__init__(dag_sink, search)
        assert fixed_operator_name in self.all_operator_names, (
            f"Operator '{fixed_operator_name}' not found. " f"Available: {self.all_operator_names}"
        )
        self.fixed_operator_name = fixed_operator_name

    def select_operators_to_evolve(self, trial: int) -> list[str]:
        assert trial > 0
        num_operators = len(self.all_operator_names)

        if trial <= num_operators:
            chosen_operator = self.all_operator_names[trial - 1]
            logger.info(f"OP-SELECTION> Initial phase: selected {chosen_operator} (trial {trial}/{num_operators})")
        else:
            chosen_operator = self.fixed_operator_name
            logger.info(f"OP-SELECTION> Fixed phase: selected {chosen_operator}")

        if _potentially_breaks_consumers(chosen_operator, self.dag_sink):
            operators_to_evolve = [chosen_operator] + self.operator_dependencies[chosen_operator]
            logger.info(
                f"OP-SELECTION> Selected {chosen_operator} with dependents: {self.operator_dependencies[chosen_operator]}"
            )
        else:
            operators_to_evolve = [chosen_operator]
            logger.info(f"OP-SELECTION> Selected {chosen_operator} without dependents")

        return operators_to_evolve
