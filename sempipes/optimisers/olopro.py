import numpy as np
import skrub
from skrub import DataOp

from sempipes.optimisers.pipeline_summary import summarise_pipeline
from sempipes.optimisers.search_strategy import GreedySearch, Outcome, SearchStrategy


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


def optimise_olopro(
    dag_sink: DataOp,
    operator_name: str,
    budget: int,
    search: SearchStrategy = GreedySearch(),
    scoring: str = "accuracy",
    cv: int = 3,
) -> list[Outcome]:
    """
    Optimises a single semantic operator in a pipeline with "operator-local" OPRO.
    """

    print("--- COMPUTING PIPELINE SUMMARY for context-aware optimisation ---")
    pipeline_summary = summarise_pipeline(dag_sink)

    search.initialize_search(dag_sink, operator_name)

    for trial in range(budget):
        search_node = search.next_search_node()

        print(f"### Processing trial {trial}")

        print("  --- Fitting pipeline")
        pipeline = dag_sink.skb.make_learner(fitted=False)
        env = _env_for_fit(dag_sink, operator_name, search_node, pipeline_summary)
        pipeline.fit(env)

        op_state = search.record_fit(pipeline, operator_name)

        print("  --- Evaluating pipeline via cross-validation")
        env = _env_for_evaluation(dag_sink, operator_name, op_state, pipeline_summary)
        cv_results = skrub.cross_validate(pipeline, env, cv=cv, scoring=scoring)
        score = np.mean(cv_results["test_score"])
        print(f"  --- Score changed to {score}")

        search.record_score(score)

    return search.get_outcomes()
