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
from sempipes.operators.operators import OptimisableMixin
from sempipes.operators.sem_choose_llm import SemChooseLLM
from sempipes.optimisers.greedy_tree_search import TreeSearch
from sempipes.optimisers.search_policy import Outcome, SearchNode, SearchPolicy
from sempipes.optimisers.trajectory import Trajectory, save_trajectory_as_json, serialize_scoring

logger = get_logger()


def _update_sem_choices(pipeline, previous_results):
    estimator_op_names = []
    choice_op_names = []

    all_choices = choice_graph(pipeline)
    for display_name in all_choices["choice_display_names"].values():
        if display_name.startswith("__sempipes__"):
            estimator_name = display_name.split("__")[2]
            estimator_op_name = f"sempipes__choices__{estimator_name}__estimator"
            choice_op_name = f"sempipes__choices__{estimator_name}__choices"
            if estimator_op_name not in estimator_op_names:
                estimator_op_names.append(estimator_op_name)
                choice_op_names.append(choice_op_name)

    for estimator_op_name, choice_op_name in zip(estimator_op_names, choice_op_names):
        apply_op = find_node_by_name(pipeline, estimator_op_name)
        choice_storage_op = find_node_by_name(pipeline, choice_op_name)

        estimator = apply_op._skrub_impl.estimator
        choices = choice_storage_op.skb.eval()

        SemChooseLLM().set_params_on_estimator(estimator, choices, previous_results=previous_results)


def optimise_sem_choices(
    dag_sink: DataOp,
    budget: int,
    scoring: str = "accuracy",
    cv: int = 3,
    candidates_to_evaluate_per_trial: int = 10,
) -> list[Outcome]:
    results = []

    logger.info("COLOPRO> Running initial search for sem_choose")
    initial_search = dag_sink.skb.make_randomized_search(
        fitted=True, cv=cv, scoring=scoring, n_iter=candidates_to_evaluate_per_trial
    )

    results.append(initial_search.results_)

    for trial in range(1, budget):
        logger.info(f"COLOPRO> Running search {trial} for sem_choose")
        _update_sem_choices(dag_sink, results)
        search = dag_sink.skb.make_randomized_search(
            fitted=True, cv=cv, scoring=scoring, n_iter=candidates_to_evaluate_per_trial
        )
        results.append(search.results_)

    return results


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
    operator_name: str,
    num_trials: int,
    search: SearchPolicy = TreeSearch(),
    scoring: str = "accuracy",
    cv=5,
    num_hpo_iterations_per_trial: int = 10,
    pipeline_definition: Callable[..., DataOp] | None = None,
    run_name: str | None = None,
    additional_env_variables: dict[str, Any] | None = None,
    n_jobs_for_evaluation: int = -1,
) -> list[Outcome]:
    """
    Optimises a single semantic operator in a pipeline with "operator-local" OPRO.
    """

    env_for_evolution = dag_sink.skb.get_data()
    env_for_scoring = dag_sink.skb.get_data()

    if additional_env_variables is not None:
        env_for_evolution.update(additional_env_variables)
        env_for_scoring.update(additional_env_variables)

    needs_hpo = _needs_hpo(dag_sink)

    logger.info("COLOPRO> Computing pipeline summary for context-aware optimisation")
    pipeline_summary = summarise_pipeline(dag_sink, pipeline_definition)
    pipeline_summary.target_metric = scoring

    search_node_queue: list[SearchNode] = []

    for trial in range(num_trials):
        logger.info(f"COLOPRO> Processing trial {trial}")

        if trial == 0:
            logger.info(f"COLOPRO> Initialising optimisation of {operator_name} via OPRO")
            search_node = search.create_root_node(dag_sink, operator_name)
            operator_state = search_node.predefined_state
            operator_memory_update = OptimisableMixin.EMPTY_MEMORY_UPDATE
        else:
            if len(search_node_queue) == 0:
                next_search_node = search.create_next_search_node()
                if next_search_node is None:
                    logger.warning("COLOPRO> Search policy returned None, no more nodes to generate")
                    break
                logger.info("COLOPRO> Generating new search node")
                search_node_queue.append(next_search_node)

            search_node = search_node_queue.pop(0)
            search_node.trial = trial
            logger.info(f'COLOPRO> Evolving operator "{operator_name}" via OPRO')
            evolution_start_time = time.time()
            pipeline = dag_sink.skb.clone()

            env_for_evolution[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
            env_for_evolution[f"sempipes_memory__{operator_name}"] = search_node.memory
            env_for_evolution[f"sempipes_inspirations__{operator_name}"] = search_node.inspirations

            operator_state, operator_memory_update = _evolve_operator(pipeline, operator_name, env_for_evolution)
            evolution_end_time = time.time()
            logger.info(f"COLOPRO> Evolution took {evolution_end_time - evolution_start_time:.2f} seconds")

        env_for_scoring[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
        env_for_scoring[f"sempipes_prefitted_state__{operator_name}"] = operator_state

        evaluation_start_time = time.time()
        if needs_hpo:
            logger.info(f"COLOPRO> Evaluating pipeline via {cv}-fold cross-validation and random search HPO")
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
            logger.info(f"COLOPRO> Evaluating pipeline via {cv}-fold cross-validation")
            pipeline = dag_sink.skb.make_learner(fitted=False)
            cv_results = skrub.cross_validate(
                pipeline, env_for_scoring, cv=cv, scoring=scoring, n_jobs=n_jobs_for_evaluation
            )
            score = float(np.mean(cv_results["test_score"]))
        evaluation_end_time = time.time()
        logger.info(f"COLOPRO> Pipeline evaluation took {evaluation_end_time - evaluation_start_time:.2f} seconds")

        logger.info(f"COLOPRO> Score changed from {search_node.parent_score} to {score}")
        search.record_outcome(search_node, operator_state, score, operator_memory_update)  # type: ignore[arg-type]

    trajectory = Trajectory(
        sempipes_config=get_config(),
        optimizer_args={
            "operator_name": operator_name,
            "num_trials": num_trials,
            "scoring": serialize_scoring(scoring),
            # "scoring": scoring if not isinstance(scoring, Callable) else f"<function: {scoring.__name__}>" if hasattr(scoring, "__name__") else f"<callable: {type(scoring).__name__}>",
            "cv": str(cv),
            "num_hpo_iterations_per_trial": num_hpo_iterations_per_trial,
        },
        outcomes=search.get_outcomes(),
    )

    trajectory_output_path = save_trajectory_as_json(trajectory, run_name=run_name)
    logger.info(f"COLOPRO> Saved trajectory to {trajectory_output_path}")

    return search.get_outcomes()
