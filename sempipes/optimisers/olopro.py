import time

import numpy as np
import skrub
from skrub import DataOp
from skrub._data_ops._evaluation import choice_graph, find_node_by_name

from sempipes.inspection.pipeline_summary import summarise_pipeline
from sempipes.operators.operators import OptimisableMixin
from sempipes.operators.sem_choose_llm import SemChooseLLM
from sempipes.optimisers.search_policy import Outcome, SearchPolicy, TreeSearch


def _env_for_fit(dag_sink, operator_name, search_node, pipeline_summary):
    assert search_node.predefined_state is None

    env = dag_sink.skb.get_data()
    env[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
    env[f"sempipes_memory__{operator_name}"] = search_node.memory

    return env


def _env_for_evaluation(dag_sink, operator_name, operator_state, pipeline_summary):
    env = dag_sink.skb.get_data()
    env[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
    env[f"sempipes_prefitted_state__{operator_name}"] = operator_state

    return env


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

    print("\tOLOPRO> Running initial search for sem_choose ---")
    initial_search = dag_sink.skb.make_randomized_search(
        fitted=True, cv=cv, scoring=scoring, n_iter=candidates_to_evaluate_per_trial
    )

    results.append(initial_search.results_)

    for trial in range(1, budget):
        print(f"\tOLOPRO> Running search {trial} for sem_choose ---")
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


def optimise_olopro(  # pylint: disable=too-many-positional-arguments, too-many-locals
    dag_sink: DataOp,
    operator_name: str,
    num_trials: int,
    search: SearchPolicy = TreeSearch(),
    scoring: str = "accuracy",
    cv: int = 3,
    num_hpo_iterations_per_trial: int = 10,
) -> list[Outcome]:
    """
    Optimises a single semantic operator in a pipeline with "operator-local" OPRO.
    """
    needs_hpo = _needs_hpo(dag_sink)

    print("\tOLOPRO> Computing pipeline summary for context-aware optimisation ---")
    pipeline_summary = summarise_pipeline(dag_sink)

    for trial in range(num_trials):
        print(f"\tOLOPRO> Processing trial {trial}")

        if trial == 0:
            print(f"\tOLOPRO> Initialising optimisation of {operator_name} via OPRO")
            search_node = search.create_root_node(dag_sink, operator_name)
            operator_state = search_node.predefined_state
            operator_memory_update = OptimisableMixin.EMPTY_MEMORY_UPDATE
        else:
            print(f'\tOLOPRO> Evolving operator "{operator_name}" via OPRO')
            evolution_start_time = time.time()
            search_node = search.create_next_search_node()
            pipeline = dag_sink.skb.clone()
            env = _env_for_fit(dag_sink, operator_name, search_node, pipeline_summary)
            operator_state, operator_memory_update = _evolve_operator(pipeline, operator_name, env)
            evolution_end_time = time.time()
            print(f"\tOLOPRO> Evolution took {evolution_end_time - evolution_start_time:.2f} seconds")

        env = _env_for_evaluation(dag_sink, operator_name, operator_state, pipeline_summary)
        evaluation_start_time = time.time()
        if needs_hpo:
            print(f"\tOLOPRO> Evaluating pipeline via {cv}-fold cross-validation and random search HPO")
            hpo = dag_sink.skb.make_randomized_search(
                fitted=False,
                cv=cv,
                scoring=scoring,
                n_iter=num_hpo_iterations_per_trial,
                n_jobs=-1,
            )
            hpo.fit(env)
            print("\tOLOPRO> " + str(hpo.results_).replace("\n", "\n\tOLOPRO> "))
            index_of_row_with_max_score = hpo.results_["mean_test_score"].idxmax()
            row_with_max_score = hpo.results_.loc[index_of_row_with_max_score]
            score = row_with_max_score["mean_test_score"]
        else:
            print(f"\tOLOPRO> Evaluating pipeline via {cv}-fold cross-validation")
            pipeline = dag_sink.skb.make_learner(fitted=False)
            cv_results = skrub.cross_validate(pipeline, env, cv=cv, scoring=scoring, n_jobs=-1)
            score = float(np.mean(cv_results["test_score"]))
        evaluation_end_time = time.time()
        print(f"\tOLOPRO> Pipeline evaluation took {evaluation_end_time - evaluation_start_time:.2f} seconds")

        print(f"\tOLOPRO> Score changed from {search_node.parent_score} to {score}")
        search.record_outcome(search_node, operator_state, score, operator_memory_update)

    return search.get_outcomes()
