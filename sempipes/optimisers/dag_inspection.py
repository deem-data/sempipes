from skrub._data_ops._evaluation import DataOp, graph

_OPERATOR_NAME_PREFIX = "sempipes_prefitted_state__"

def extract_operator_names(pipeline):
    env_for_inspection = pipeline.skb.get_data()
    all_operator_names = [
        name.split("__")[1] for name in env_for_inspection.keys() if name.startswith(_OPERATOR_NAME_PREFIX)
    ]
    all_operator_names = sorted(list(set(all_operator_names)))
    return all_operator_names


def _hop(current_node_id, dag, candidate_operator_names, dependent_operator_names):
    if current_node_id in dag['parents']:
        children_ids = dag['parents'][current_node_id]
        for child_id in children_ids:
            child_node = dag['nodes'][child_id]
            if isinstance(child_node, DataOp) and child_node._skrub_impl.name in candidate_operator_names:
                dependent_operator_names.append(child_node._skrub_impl.name)                     
            _hop(child_id, dag, candidate_operator_names, dependent_operator_names)

# TODO: This code needs to check the operator types / configs, not each case requires dependent operators
def collect_dependent_operators(pipeline):

    dag = graph(pipeline)
    all_operator_names = extract_operator_names(pipeline)

    operator_dependencies = {}
    for op_of_interest_name in all_operator_names:
        op_id = next((node_id for node_id, node in dag['nodes'].items() if isinstance(node, DataOp) and node._skrub_impl.name == op_of_interest_name), None)
        assert op_id is not None
        candidate_operator_names = [name for name in all_operator_names if name != op_of_interest_name]
        dependent_operator_names = []
        _hop(op_id, dag, candidate_operator_names, dependent_operator_names)
        unique_dependent_operator_names = list(dict.fromkeys(dependent_operator_names))
        operator_dependencies[op_of_interest_name] = unique_dependent_operator_names

    return operator_dependencies   
