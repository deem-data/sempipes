from IPython.display import Code
from skrub._data_ops._evaluation import find_node_by_name


# TODO This code only works for certain operators and should do some null checking and error handling.
def inspect_generated_code(pipeline, operator_name, eval_mode="preview"):
    operator_of_interest = find_node_by_name(pipeline, f"sempipes_fitted_estimator__{operator_name}")
    code = operator_of_interest.generated_code_._skrub_impl.results[eval_mode]

    if isinstance(code, list):
        code = "\n".join(code)

    return Code(code, language="python")
