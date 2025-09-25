import inspect
import traceback

from pandas import DataFrame
from sklearn.base import BaseEstimator

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.inspection.runtime_summary import available_packages
from sempipes.llm.llm import generate_python_code
from sempipes.operators.operators import SemChoices, SemChooseOperator

_MAX_RETRIES = 3


def _default_of(estimator_cls, param_name):
    signature = inspect.signature(estimator_cls.__init__)
    param = signature.parameters.get(param_name)
    assert param is not None

    if param.default is inspect._empty:
        return None
    return param.default


class SemChooseLLM(SemChooseOperator):
    def set_params_on_estimator(
        self, estimator: BaseEstimator, choices: SemChoices, previous_results: list[DataFrame] | None = None
    ) -> None:
        print(f"--- sempipes.apply_with_sem_choose({estimator}, {choices})")

        choice_name = choices.name
        for param_name, user_prompt in choices.params_and_prompts.items():
            previous_exceptions: list[str] = []
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    prompt = self._build_prompt(
                        estimator,
                        user_prompt,
                        choices.name,
                        param_name,
                        previous_results=previous_results,
                        previous_exceptions=previous_exceptions,
                    )

                    python_code = generate_python_code(prompt)

                    suggested_choices = safe_exec(python_code, "__generated_sempipes_choices")
                    suggested_choices.name = f"__sempipes__{choice_name}__{param_name}"
                    estimator.set_params(**{param_name: suggested_choices})
                    print(f"\tSuggested choices for {param_name}: {suggested_choices}")
                    break
                except Exception as e:  # pylint: disable=broad-except
                    print(f"An error occurred in attempt {attempt}:", e)
                    tb_str = traceback.format_exc()
                    previous_exceptions.append(tb_str)

    @staticmethod
    def _build_prompt(
        estimator: BaseEstimator,
        user_prompt: str,
        choice_name: str,
        param_name: str,
        previous_results: list[DataFrame] | None,
        previous_exceptions: list[str] | None = None,
    ) -> str:
        default_hint = ""
        default_value_for_param = _default_of(estimator.__class__, param_name)
        if default_value_for_param is not None:
            default_hint = f"Note that the default value for this parameter is {default_value_for_param!r}, which may be a reasonable choice to include."

        previous_results_log = ""
        if previous_results is not None and len(previous_results) > 0:
            previous_results_log = """
            # HISTORICAL INFORMATION ABOUT PREVIOUS EVALUATIONS:
            Several hyperparameter choices have already been evaluated, here is a log what happened so far. Please use this information to avoid suggesting choices that have already been tried and did not work well.\n
            """

            for run, previous_result in enumerate(previous_results):
                hyperparam_columns = [
                    column.replace(f"__sempipes__{choice_name}__", "")
                    for column in previous_result.columns
                    if column.startswith(f"__sempipes__{choice_name}__")
                ]

                for _, row in previous_result.iterrows():
                    setting = " ".join(
                        [
                            f"{hyperparam_column}={row[f'__sempipes__{choice_name}__' + hyperparam_column]}"
                            for hyperparam_column in hyperparam_columns
                        ]
                    )
                    previous_results_log += (
                        f"Evaluation {run} with {setting} resulted in a test score of {row['mean_test_score']}\n"
                    )

        previous_exceptions_memory = ""
        if previous_exceptions is None:
            previous_exceptions = []

        if len(previous_exceptions) > 0:
            previous_exceptions_memory = """
            # PREVIOUS FAILURES:
            This request has been previously attempted, but the generated code did not work. 
            Here are the previous exceptions:\n
            """
            previous_exceptions_memory += "\n".join([f"### Previous exception: {str(e)}" for e in previous_exceptions])

        estimator_name = f"{estimator.__class__.__module__}.{estimator.__class__.__name__}"

        available_packages_hint = ", ".join(
            [f"{package_name}=={package_version}" for package_name, package_version in available_packages().items()]
        )

        return f"""
        You need to help a data scientist improve their machine learning script in Python, which is written 
        using Pandas, sklearn and skrub. They are using {available_packages_hint}.
    
        # HERE IS AN EXAMPLE:
    
        The data scientist needs help with choosing the `penalty` of a `sklearn.linear_model.LogisticRegression`. 
        Their specific request is:
    
        ```Regularizers that work well for high-dimensional data.```  
    
        This is an example of a valid response:
    
        __generated_sempipes_choices = skrub.choose_from(['l1', 'elasticnet'])
    
        # HERE IS ANOTHER EXAMPLE:
    
        The data scientist needs help with choosing the `high_cardinality` of a `skrub.TableVectorizer`. 
        Their specific request is:
    
        ```Non-embedding based techniques to encode text data.```  
    
        This is an example of a valid response:
    
        __generated_sempipes_choices = skrub.choose_from([
            skrub.MinHashEncoder(n_components=30, ngram_range=(2, 4)),
            skrub.GapEncoder(n_components=10, batch_size=1024)
        ])
    
        {previous_exceptions_memory}
    
        # HERE COMES THE ACTUAL REQUEST:
    
        The data scientist needs help with choosing the `{param_name}` of a `{estimator_name}`. 
        Their specific request is:
    
        ```{user_prompt}```
    
        {default_hint}
    
        Please respond with Python code that specifies parameter values that match their request. Make sure that 
        the Python code is executable and uses the correct types. IMPORTANT: The first suggested choice should be 
        the most likely to work well, and is very critical to get right since it will often be used as a default value! 
    
        {previous_results_log}
    
        ONLY INCLUDE VALID PYTHON CODE IN YOUR RESPONSE, NO MARKDOWN OR TEXT. 
        
        THE PYTHON CODE SHOULD MATCH THIS TEMPLATE:
    
        __generated_sempipes_choices = skrub.choose_from([<Your suggested parameter values>])
        """
