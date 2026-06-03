import time
from collections.abc import Callable
from typing import Any

import numpy as np
import skrub
from skrub import DataOp
from skrub._data_ops._evaluation import choice_graph, find_node_by_name

from sempipes import get_config
from sempipes.inspection.pipeline_summary import summarise_pipeline
from sempipes.logging import get_logger
from sempipes.optimisers.greedy_tree_search import TreeSearch
from sempipes.optimisers.operator_selection import (
    AdaEvolveGlobalNormOperatorSelectionPolicy,
    AllOperatorsPolicy,
    FixedOpPolicy,
    OperatorSelectionPolicy,
    RoundRobinOperatorSelectionPolicy,
    UCBOperatorSelectionPolicy,
)
from sempipes.optimisers.search_policy import Outcome, SearchPolicy
from sempipes.optimisers.trajectory import Trajectory, save_trajectory_as_json, serialize_scoring

logger = get_logger()


def _evolve_operator(pipeline, operator_name, env):
    operator_to_recompute = find_node_by_name(pipeline, operator_name)
    operator_to_recompute.skb.eval(env)
    fitted = operator_to_recompute._skrub_impl.estimator_
    operator_state = fitted.state_after_fit()
    operator_memory_update = fitted.memory_update_from_latest_fit()
    return operator_state, operator_memory_update


def _needs_hpo(dag_sink):
    pipeline_choices = choice_graph(dag_sink)
    return len(pipeline_choices["choices"]) > 0


def optimise_colopro(  # pylint: disable=too-many-positional-arguments, too-many-locals, too-many-statements, too-many-arguments
    dag_sink: DataOp,
    num_trials: int,
    search: SearchPolicy = TreeSearch(),
    scoring: str = "accuracy",
    cv=5,
    num_hpo_iterations_per_trial: int = 10,
    pipeline_definition: Callable[..., DataOp] | None = None,
    run_name: str | None = None,
    additional_env_variables: dict[str, Any] | None = None,
    n_jobs_for_evaluation: int = -1,
    only_optimize_operator: str | None = None,
    optimize_all_operators: bool = False,
    operator_selection_strategy: str = "ucb",
) -> list[Outcome]:
    """
    Optimises a single semantic operator in a pipeline with "operator-local" OPRO.
    """

    operator_selection_policy: OperatorSelectionPolicy
    logger.info(f"COLOPRO> Operator selection strategy: {operator_selection_strategy}")
    if optimize_all_operators:
        operator_selection_policy = AllOperatorsPolicy(dag_sink, search)
    elif only_optimize_operator is None:
        if operator_selection_strategy == "ucb":
            operator_selection_policy = UCBOperatorSelectionPolicy(dag_sink, search)
        elif operator_selection_strategy == "adaevolve_global_norm":
            operator_selection_policy = AdaEvolveGlobalNormOperatorSelectionPolicy(dag_sink, search)
        elif operator_selection_strategy == "round_robin":
            operator_selection_policy = RoundRobinOperatorSelectionPolicy(dag_sink, search)
        else:
            raise ValueError(
                "Unknown operator_selection_strategy "
                f"'{operator_selection_strategy}'. Expected one of: "
                "'ucb', 'adaevolve_global_norm', 'round_robin'."
            )
    else:
        operator_selection_policy = FixedOpPolicy(dag_sink, search, only_optimize_operator)

    needs_hpo = _needs_hpo(dag_sink)

    logger.info("COLOPRO> Computing pipeline summary for context-aware optimisation")
    pipeline_summary = summarise_pipeline(dag_sink, pipeline_definition)
    pipeline_summary.target_metric = scoring

    for trial in range(num_trials):
        logger.info(f"COLOPRO> Processing trial {trial}")
        states_of_operators_to_evolve: dict[str, dict[str, Any]] = {}
        memory_updates_of_operators_to_evolve: dict[str, str] = {}
        env_for_scoring = dag_sink.skb.get_data()

        if additional_env_variables is not None:
            env_for_scoring.update(additional_env_variables)

        if trial == 0:
            logger.info("COLOPRO> Initialising optimisation via OPRO")
            search_node = search.create_root_node(dag_sink, operator_selection_policy.all_operator_names)
        else:
            operators_to_evolve = operator_selection_policy.select_operators_to_evolve(trial)
            chosen_operator = operators_to_evolve[0]

            search_node = search.create_next_search_node(
                trial, chosen_operator, operator_selection_policy.all_operator_names
            )
            logger.info("COLOPRO> Asking search policy to generate next search node")

            evolution_start_time = time.time()

            for operator_to_evolve in operators_to_evolve:
                logger.info(f'COLOPRO> OP_EVOLUTION> Evolving operator "{operator_to_evolve}" via OPRO')

                pipeline = dag_sink.skb.clone()
                env_for_evolution = dag_sink.skb.get_data()
                if additional_env_variables is not None:
                    env_for_evolution.update(additional_env_variables)

                env_for_evolution[f"sempipes_pipeline_summary__{operator_to_evolve}"] = pipeline_summary
                env_for_evolution[f"sempipes_memory__{operator_to_evolve}"] = search_node.memories.get(
                    operator_to_evolve
                )
                env_for_evolution[f"sempipes_inspirations__{operator_to_evolve}"] = search_node.inspirations.get(
                    operator_to_evolve
                )

                fixed_operators = [
                    name for name in operator_selection_policy.all_operator_names if name != operator_to_evolve
                ]
                for fixed_operator in fixed_operators:
                    fixed_state = states_of_operators_to_evolve.get(fixed_operator) or search_node.fixed_states.get(
                        fixed_operator
                    )
                    env_for_evolution[f"sempipes_prefitted_state__{fixed_operator}"] = fixed_state

                operator_state, operator_memory_update = _evolve_operator(
                    pipeline, operator_to_evolve, env_for_evolution
                )

                logger.info(f"COLOPRO> OP_EVOLUTION> {operator_to_evolve} evolved...")
                states_of_operators_to_evolve[operator_to_evolve] = operator_state
                memory_updates_of_operators_to_evolve[operator_to_evolve] = operator_memory_update

            search_node.operators_evolved = operators_to_evolve
            evolution_end_time = time.time()
            logger.info(f"COLOPRO> Evolution took {evolution_end_time - evolution_start_time:.2f} seconds")

        scoring_start_time = time.time()
        for fixed_operator in operator_selection_policy.all_operator_names:
            if fixed_operator in states_of_operators_to_evolve:
                env_for_scoring[f"sempipes_prefitted_state__{fixed_operator}"] = states_of_operators_to_evolve[
                    fixed_operator
                ]
            else:
                env_for_scoring[f"sempipes_prefitted_state__{fixed_operator}"] = search_node.fixed_states.get(
                    fixed_operator
                )

        if needs_hpo:
            logger.info(f"COLOPRO> Scoring pipeline via {cv}-fold cross-validation and random search HPO")
            hpo = dag_sink.skb.make_randomized_search(
                fitted=False,
                cv=cv,
                scoring=scoring,
                n_iter=num_hpo_iterations_per_trial,
                n_jobs=-1,
            )
            hpo.fit(env_for_scoring)
            index_of_row_with_max_score = hpo.results_["mean_test_score"].idxmax()
            row_with_max_score = hpo.results_.loc[index_of_row_with_max_score]
            score = row_with_max_score["mean_test_score"]
        else:
            logger.info(f"COLOPRO> Scoring pipeline via {cv}-fold cross-validation")
            pipeline = dag_sink.skb.make_learner(fitted=False)
            cv_results = skrub.cross_validate(
                pipeline, env_for_scoring, cv=cv, scoring=scoring, n_jobs=n_jobs_for_evaluation
            )
            score = float(np.mean(cv_results["test_score"]))
        scoring_end_time = time.time()
        logger.info(f"COLOPRO> Pipeline scoring took {scoring_end_time - scoring_start_time:.2f} seconds")

        logger.info(f"COLOPRO> Score changed from {search_node.parent_score} to {score}")
        search.record_outcome(search_node, states_of_operators_to_evolve, score, memory_updates_of_operators_to_evolve)

    trajectory = Trajectory(
        sempipes_config=get_config(),
        optimizer_args={
            "num_trials": num_trials,
            "operator_selection_policy": operator_selection_policy.as_dict(),
            "scoring": serialize_scoring(scoring),
            "cv": str(cv),
            "num_hpo_iterations_per_trial": num_hpo_iterations_per_trial,
            "operator_selection_strategy": operator_selection_strategy,
        },
        outcomes=search.get_outcomes(),
    )

    trajectory_output_path = save_trajectory_as_json(trajectory, run_name=run_name)
    logger.info(f"COLOPRO> Saved trajectory to {trajectory_output_path}")

    return search.get_outcomes()
