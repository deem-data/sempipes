from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from skrub import DataOp
from skrub._data_ops._data_ops import Apply, GetItem
from skrub._data_ops._evaluation import find_node, find_X, find_y


@dataclass
class DagSummary:
    task_type: str | None = None
    model: str | None = None
    model_name: str | None = None
    model_steps: str | None = None
    model_definition: str | None = None
    target_name: str | None = None
    target_definition: str | None = None
    target_steps: str | None = None
    target_unique_values_from_preview: list[Any] | None = None
    dataset_description: str | None = None


# TODO We can do much more here!
def _summarise_dag(dag_sink_node: DataOp) -> DagSummary:
    summary = DagSummary()

    def is_model(some_op):
        if hasattr(some_op, "_skrub_impl"):
            impl = some_op._skrub_impl
            if isinstance(impl, Apply) and hasattr(impl, "estimator"):
                est = impl.estimator
                return isinstance(est, (ClassifierMixin, RegressorMixin))
        return False

    model_node = find_node(dag_sink_node, predicate=is_model)
    if model_node is not None:
        estimator = model_node._skrub_impl.estimator
        summary.model_steps = model_node.skb.describe_steps()
        summary.model_definition = model_node._skrub_impl.creation_stack_description()
        summary.model = f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__}"
        if isinstance(estimator, ClassifierMixin):
            summary.task_type = "classification"
        if isinstance(estimator, RegressorMixin):
            summary.task_type = "regression"

    y_op = find_y(dag_sink_node)
    if y_op is not None:
        summary.target_steps = y_op.skb.describe_steps()
        if hasattr(y_op, "_skrub_impl"):
            summary.target_definition = y_op._skrub_impl.creation_stack_description()
            if y_op.skb.name is not None:
                summary.target_name = y_op.skb.name
            elif isinstance(y_op._skrub_impl, GetItem):
                summary.target_name = y_op._skrub_impl.key
            try:
                summary.target_unique_values_from_preview = [val.item() for val in np.unique(y_op.skb.preview())]
            except Exception as __e:  # pylint: disable=broad-exception-caught
                pass

    X_op = find_X(dag_sink_node)
    if X_op is not None:
        if X_op.skb.description is not None:
            summary.dataset_description = X_op.skb.description

    return summary
