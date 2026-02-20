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
from sempipes.optimisers.greedy_tree_search import TreeSearch
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
) -> list[Outcome]:
    """
    Optimises a single semantic operator in a pipeline with "operator-local" OPRO.
    """
    env_for_inspection = dag_sink.skb.get_data()
    # collect all operator names from the environment, the variable name is "sempipes_prefitted_state__{name}"
    all_operator_names = [
        name.split("__")[1] for name in env_for_inspection.keys() if name.startswith("sempipes_prefitted_state__")
    ]

    all_operator_names = sorted(list(set(all_operator_names)))

    needs_hpo = _needs_hpo(dag_sink)

    logger.info("COLOPRO> Computing pipeline summary for context-aware optimisation")
    pipeline_summary = summarise_pipeline(dag_sink, pipeline_definition)
    pipeline_summary.target_metric = scoring

    for trial in range(num_trials):
        logger.info(f"COLOPRO> Processing trial {trial}")

        env_for_evolution = dag_sink.skb.get_data()
        env_for_scoring = dag_sink.skb.get_data()

        if additional_env_variables is not None:
            env_for_evolution.update(additional_env_variables)
            env_for_scoring.update(additional_env_variables)

        if trial == 0:
            logger.info("COLOPRO> Initialising optimisation via OPRO")
            search_node = search.create_root_node(dag_sink, all_operator_names)
            operator_state = None
            fixed_operators = all_operator_names
            operator_memory_update = OptimisableMixin.EMPTY_MEMORY_UPDATE
        else:
            # Pick the operator to evolve from all_operator_names in round-robin fashion
            operator_to_evolve = all_operator_names[trial % len(all_operator_names)]
            fixed_operators = [name for name in all_operator_names if name != operator_to_evolve]

            search_node = search.create_next_search_node(trial, operator_to_evolve, all_operator_names)
            logger.info("COLOPRO> Generating new search node")

            logger.info(f'COLOPRO> Evolving operator "{operator_to_evolve}" via OPRO')
            evolution_start_time = time.time()
            pipeline = dag_sink.skb.clone()

            env_for_evolution[f"sempipes_pipeline_summary__{operator_to_evolve}"] = pipeline_summary
            env_for_evolution[f"sempipes_memory__{operator_to_evolve}"] = search_node.memories.get(operator_to_evolve)
            env_for_evolution[f"sempipes_inspirations__{operator_to_evolve}"] = search_node.inspirations.get(
                operator_to_evolve
            )

            for operator_name in fixed_operators:
                env_for_evolution[f"sempipes_prefitted_state__{operator_name}"] = search_node.fixed_states.get(
                    operator_name
                )

            operator_state, operator_memory_update = _evolve_operator(pipeline, operator_to_evolve, env_for_evolution)
            evolution_end_time = time.time()
            logger.info(f"COLOPRO> Evolution took {evolution_end_time - evolution_start_time:.2f} seconds")

            env_for_scoring[f"sempipes_prefitted_state__{operator_to_evolve}"] = operator_state

        for operator_name in fixed_operators:
            env_for_scoring[f"sempipes_prefitted_state__{operator_name}"] = search_node.fixed_states.get(operator_name)

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
            # "operator_name": operator_name,
            "num_trials": num_trials,
            "scoring": serialize_scoring(scoring),
            "cv": str(cv),
            "num_hpo_iterations_per_trial": num_hpo_iterations_per_trial,
        },
        outcomes=search.get_outcomes(),
    )

    trajectory_output_path = save_trajectory_as_json(trajectory, run_name=run_name)
    logger.info(f"COLOPRO> Saved trajectory to {trajectory_output_path}")

    return search.get_outcomes()
