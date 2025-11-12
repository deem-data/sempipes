import time
from collections.abc import Callable

import numpy as np
import skrub
from skrub import DataOp
from skrub._data_ops._evaluation import choice_graph, find_node_by_name

from sempipes import get_config
from sempipes.inspection.pipeline_summary import summarise_pipeline
from sempipes.operators.operators import OptimisableMixin
from sempipes.operators.sem_choose_llm import SemChooseLLM
from sempipes.optimisers import TreeSearch
from sempipes.optimisers.search_policy import Outcome, SearchNode, SearchPolicy
from sempipes.optimisers.trajectory import Trajectory, save_trajectory_as_json


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

    print("\tCOLOPRO> Running initial search for sem_choose ---")
    initial_search = dag_sink.skb.make_randomized_search(
        fitted=True, cv=cv, scoring=scoring, n_iter=candidates_to_evaluate_per_trial
    )

    results.append(initial_search.results_)

    for trial in range(1, budget):
        print(f"\tCOLOPRO> Running search {trial} for sem_choose ---")
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


def optimise_colopro(  # pylint: disable=too-many-positional-arguments, too-many-locals
    dag_sink: DataOp,
    operator_name: str,
    num_trials: int,
    search: SearchPolicy = TreeSearch(),
    scoring: str = "accuracy",
    cv=3,
    num_hpo_iterations_per_trial: int = 10,
    pipeline_definition: Callable[..., DataOp] | None = None,
    run_name: str | None = None,
) -> list[Outcome]:
    """
    Optimises a single semantic operator in a pipeline with "operator-local" OPRO.
    """

    env_for_evolution = dag_sink.skb.get_data()
    env_for_scoring = dag_sink.skb.get_data()

    needs_hpo = _needs_hpo(dag_sink)

    print("\tCOLOPRO> Computing pipeline summary for context-aware optimisation ---")
    pipeline_summary = summarise_pipeline(dag_sink, pipeline_definition)
    pipeline_summary.target_metric = scoring

    search_node_queue: list[SearchNode] = []

    for trial in range(num_trials):
        print(f"\tCOLOPRO> Processing trial {trial}")

        if trial == 0:
            print(f"\tCOLOPRO> Initialising optimisation of {operator_name} via OPRO")
            search_node = search.create_root_node(dag_sink, operator_name)
            operator_state = search_node.predefined_state
            operator_memory_update = OptimisableMixin.EMPTY_MEMORY_UPDATE
        else:
            if len(search_node_queue) == 0:
                next_search_nodes = search.create_next_search_nodes()
                print(f"\tCOLOPRO> Generating {len(next_search_nodes)} new search node(s)")
                search_node_queue.extend(next_search_nodes)

            search_node = search_node_queue.pop(0)
            search_node.trial = trial
            print(f'\tCOLOPRO> Evolving operator "{operator_name}" via OPRO')
            evolution_start_time = time.time()
            pipeline = dag_sink.skb.clone()

            env_for_evolution[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
            env_for_evolution[f"sempipes_memory__{operator_name}"] = search_node.memory

            operator_state, operator_memory_update = _evolve_operator(pipeline, operator_name, env_for_evolution)
            evolution_end_time = time.time()
            print(f"\tCOLOPRO> Evolution took {evolution_end_time - evolution_start_time:.2f} seconds")

        env_for_scoring[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
        env_for_scoring[f"sempipes_prefitted_state__{operator_name}"] = operator_state

        evaluation_start_time = time.time()
        if needs_hpo:
            print(f"\tCOLOPRO> Evaluating pipeline via {cv}-fold cross-validation and random search HPO")
            hpo = dag_sink.skb.make_randomized_search(
                fitted=False,
                cv=cv,
                scoring=scoring,
                n_iter=num_hpo_iterations_per_trial,
                n_jobs=-1,
            )
            hpo.fit(env_for_scoring)
            print("\tCOLOPRO> " + str(hpo.results_).replace("\n", "\n\tCOLOPRO> "))
            index_of_row_with_max_score = hpo.results_["mean_test_score"].idxmax()
            row_with_max_score = hpo.results_.loc[index_of_row_with_max_score]
            score = row_with_max_score["mean_test_score"]
        else:
            print(f"\tCOLOPRO> Evaluating pipeline via {cv}-fold cross-validation")
            pipeline = dag_sink.skb.make_learner(fitted=False)
            cv_results = skrub.cross_validate(pipeline, env_for_scoring, cv=cv, scoring=scoring, n_jobs=-1)
            score = float(np.mean(cv_results["test_score"]))
        evaluation_end_time = time.time()
        print(f"\tCOLOPRO> Pipeline evaluation took {evaluation_end_time - evaluation_start_time:.2f} seconds")

        print(f"\tCOLOPRO> Score changed from {search_node.parent_score} to {score}")
        search.record_outcome(search_node, operator_state, score, operator_memory_update)  # type: ignore[arg-type]

    trajectory = Trajectory(
        sempipes_config=get_config(),
        optimizer_args={
            "operator_name": operator_name,
            "num_trials": num_trials,
            "scoring": scoring,
            "cv": str(cv),
            "num_hpo_iterations_per_trial": num_hpo_iterations_per_trial,
        },
        outcomes=search.get_outcomes(),
    )

    trajectory_output_path = save_trajectory_as_json(trajectory, run_name=run_name)
    print(f"\tCOLOPRO> Saved trajectory to {trajectory_output_path}")

    return search.get_outcomes()
