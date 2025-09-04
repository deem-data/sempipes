import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from skrub._data_ops._data_ops import Apply, GetItem
from skrub._data_ops._evaluation import find_node, find_y, find_X

# TODO We can do much more here!
def summarise_dag(dag_sink_node):
    # TODO This should be a proper dataclass
    summary = {
        "task_type": None,
        "model": None,
        "model_definition": None,
        "model_steps": None,
        "target_name": None,
        "target_definition": None,
        "target_steps": None,
        "target_unique_values_from_preview": None,
        "dataset_description": None,
    }

    def is_model(some_op):
        if hasattr(some_op, "_skrub_impl"):
            impl = some_op._skrub_impl
            if isinstance(impl, Apply) and hasattr(impl, 'estimator'):
                est = impl.estimator
                return isinstance(est, ClassifierMixin) or isinstance(est, RegressorMixin)
        return False

    model_node = find_node(dag_sink_node, predicate=lambda x: is_model(x))

    if model_node is not None:
        estimator = model_node._skrub_impl.estimator
        summary["model_steps"] = model_node.skb.describe_steps()
        summary["model_definition"] = model_node._skrub_impl.creation_stack_description()
        summary["model"] = f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__}"
        if isinstance(estimator, ClassifierMixin):
            summary["task_type"] = "classification"
        if isinstance(estimator, RegressorMixin):
            summary["task_type"] = "regression"

    y_op = find_y(dag_sink_node)

    if y_op is not None:
        summary["target_steps"] = y_op.skb.describe_steps()
        if hasattr(y_op, '_skrub_impl'):
            summary["target_definition"] = y_op._skrub_impl.creation_stack_description()
            if y_op.skb.name is not None:
                summary["target_name"] = y_op.skb.name
            elif isinstance(y_op._skrub_impl, GetItem):
                summary["target_name"] = y_op._skrub_impl.key

            try:
                summary["target_unique_values_from_preview"] = [val.item() for val in np.unique(y_op.skb.preview())]
            except:
                pass

    X_op = find_X(dag_sink_node)

    if X_op is not None:
        if X_op.skb.description is not None:
            summary["dataset_description"] = X_op.skb.description

    return summary
