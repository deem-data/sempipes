import networkx as nx
from skrub._data_ops._data_ops import DataOp, Var, GetItem, CallMethod, Call

def context_graph(op):
    graph = nx.DiGraph()
    queue = [(op, None)]
    node_id = -1

    python_id_to_node_id = {}

    while len(queue) > 0:
        current_op, parent_node_id = queue.pop(0)

        python_id = id(current_op)

        if python_id not in python_id_to_node_id:
            node_id += 1
            python_id_to_node_id[python_id] = node_id
            graph.add_node(node_id, data_op=current_op)

        else:
            node_id = python_id_to_node_id[python_id]

        if parent_node_id is not None:
            graph.add_edge(node_id, parent_node_id)

        for field_name in current_op._fields:
            field = getattr(current_op, field_name)
            if isinstance(field, DataOp):
                queue.append((field._skrub_impl, node_id))

        if isinstance(current_op, CallMethod) or isinstance(current_op, Call):
            for arg in current_op.args:
                if isinstance(arg, DataOp):
                    queue.append((arg._skrub_impl, node_id))

    return graph