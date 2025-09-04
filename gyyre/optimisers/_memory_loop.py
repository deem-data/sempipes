import skrub
import numpy as np
from gyyre.optimisers._dag_summary import summarise_dag

def optimise_semantic_operator(dag_sink, operator_name, num_iterations):
    memory = []
    states = []

    print(f"--- COMPUTING DAG SUMMARY for context-aware optimisation ---")
    dag_summary = summarise_dag(dag_sink)

    for key, value in dag_summary.items():
        if "_steps" in key and value is not None:
            value = value.replace("\n", " > ")
        if "_definition" in key and value is not None:
            value = value.replace("\n", " ")

        print(f"\t> {key}: {value}")
    print("\n")

    for iteration in range(num_iterations):
        print(f"---ITERATION {iteration} -> Fitting with memory ---")
        learner = dag_sink.skb.make_learner(fitted=False)

        env = dag_sink.skb.get_data()
        env[f'gyyre_dag_summary__{operator_name}'] = dag_summary
        env[f'gyyre_memory__{operator_name}'] = memory

        learner.fit(env)

        print(f"---ITERATION {iteration} -> Evaluation ---")
        op_fitted = learner.find_fitted_estimator(operator_name).transformer_
        op_state = op_fitted.state_after_fit()
        states.append(op_state)

        env = dag_sink.skb.get_data()
        env[f'gyyre_prefitted_state__{operator_name}'] = op_state

        op_memory_update = op_fitted.memory_update_from_latest_fit()

        cv_results = skrub.cross_validate(learner, env, cv=3)
        mean_accuracy = np.mean(cv_results['test_score'])

        memory.append({
            "update": op_memory_update,
            "accuracy": float(mean_accuracy),
        })

        print(f"---ITERATION {iteration} -> Accuracy: {mean_accuracy}")
    return memory, states