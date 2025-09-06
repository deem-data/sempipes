from typing import Any

import numpy as np
import skrub
from skrub import DataOp

from gyyre.optimisers._dag_summary import _summarise_dag


def greedy_optimise_semantic_operator(
    dag_sink: DataOp,
    operator_name: str,
    num_iterations: int,
    scoring: str = "accuracy",
    cv: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memory = []
    states = []

    print("--- COMPUTING DAG SUMMARY for context-aware optimisation ---")
    dag_summary = _summarise_dag(dag_sink)
    # print(dag_summary)

    for iteration in range(num_iterations):
        print(f"---ITERATION {iteration} -> Fitting with memory ---")
        learner = dag_sink.skb.make_learner(fitted=False)

        env = dag_sink.skb.get_data()
        env[f"gyyre_dag_summary__{operator_name}"] = dag_summary
        env[f"gyyre_memory__{operator_name}"] = memory

        learner.fit(env)

        print(f"---ITERATION {iteration} -> Evaluation ---")
        op_fitted = learner.find_fitted_estimator(operator_name).transformer_
        op_state = op_fitted.state_after_fit()
        states.append(op_state)

        env = dag_sink.skb.get_data()
        env[f"gyyre_prefitted_state__{operator_name}"] = op_state

        op_memory_update = op_fitted.memory_update_from_latest_fit()

        cv_results = skrub.cross_validate(learner, env, cv=cv, scoring=scoring)
        mean_score = np.mean(cv_results["test_score"])

        memory.append(
            {
                "update": op_memory_update,
                "score": float(mean_score),
            }
        )

        print(f"---ITERATION {iteration} -> {scoring}: {mean_score}")
    return memory, states
