import numpy as np
import skrub
from skrub import DataOp
from skrub._data_ops._evaluation import choice_graph, find_node_by_name

from sempipes.inspection.pipeline_summary import summarise_pipeline
from sempipes.operators.sem_choose_llm import SemChooseLLM
from sempipes.optimisers.search_policy import Outcome, SearchPolicy, TreeSearch


def _env_for_fit(dag_sink, operator_name, search_node, pipeline_summary):
    env = dag_sink.skb.get_data()
    env[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
    env[f"sempipes_memory__{operator_name}"] = search_node.memory

    if search_node.predefined_state is not None:
        env[f"sempipes_prefitted_state__{operator_name}"] = search_node.predefined_state

    return env


def _env_for_evaluation(dag_sink, operator_name, op_state, pipeline_summary):
    env = dag_sink.skb.get_data()
    env[f"sempipes_pipeline_summary__{operator_name}"] = pipeline_summary
    env[f"sempipes_prefitted_state__{operator_name}"] = op_state

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


def optimise_olopro(
    dag_sink: DataOp,
    operator_name: str,
    budget: int,
    search: SearchPolicy = TreeSearch(),
    scoring: str = "accuracy",
    cv: int = 3,
) -> list[Outcome]:
    """
    Optimises a single semantic operator in a pipeline with "operator-local" OPRO.
    """

    print("\tOLOPRO> Computing pipeline summary for context-aware optimisation ---")
    pipeline_summary = summarise_pipeline(dag_sink)

    search.initialize_search(dag_sink, operator_name)

    for trial in range(budget):
        search_node = search.next_search_node()

        print(f"\tOLOPRO> Processing trial {trial}")

        print("\tOLOPRO> Fitting pipeline")
        pipeline = dag_sink.skb.make_learner(fitted=False)
        env = _env_for_fit(dag_sink, operator_name, search_node, pipeline_summary)
        pipeline.fit(env)

        op_state = search.record_fit(pipeline, operator_name)

        print(f"\tOLOPRO> Evaluating pipeline via {cv}-fold cross-validation")
        env = _env_for_evaluation(dag_sink, operator_name, op_state, pipeline_summary)
        cv_results = skrub.cross_validate(pipeline, env, cv=cv, scoring=scoring)
        score = float(np.mean(cv_results["test_score"]))
        print(f"\tOLOPRO> Score changed to {score}")
        search.record_score(score)

    return search.get_outcomes()
