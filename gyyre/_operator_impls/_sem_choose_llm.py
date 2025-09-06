import traceback
from collections.abc import Iterable

from sklearn.base import BaseEstimator
from skrub import DataOp

from gyyre._code_gen._exec import _safe_exec
from gyyre._code_gen._llm import _generate_python_code
from gyyre._operators import SemChooseOperator


class SemChooseLLM(SemChooseOperator):
    def set_params_on_estimator(
        self, data_op: DataOp, estimator: BaseEstimator, choices: dict[str, str] | None, y=None
    ) -> None:
        print(f"--- gyyre.apply_with_sem_choose({estimator}, {choices})")

        max_retries = 3

        if choices is not None:
            for param_name, user_prompt in choices.items():
                previous_exceptions = []
                for attempt in range(1, max_retries + 1):
                    try:
                        prompt = self._build_prompt(
                            estimator, user_prompt, param_name, previous_exceptions=previous_exceptions
                        )
                        python_code = _generate_python_code(prompt)

                        suggested_choices = _safe_exec(python_code, "__generated_sempipes_choices")
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
        param_name: str,
        previous_exceptions: Iterable[Exception] | None = None,
    ) -> str:
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

        return f"""
        You need to help a data scientist improve their machine learning script in Python, which is written 
        using Pandas, sklearn and skrub. They are using scikit-learn==1.7.1, pandas==2.3.2 and skrub==0.6.1 
    
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
    
        Please respond with Python code that specifies parameter values that match their request. Make sure that 
        the Python code is executable and uses the correct types. IMPORTANT: The first suggested choice should be 
        the most likely to work well, and is very critical to get right since it will often be used as a default value! 
    
        ONLY INCLUDE VALID PYTHON CODE IN YOUR RESPONSE, NO MARKDOWN OR TEXT. 
        
        THE PYTHON CODE SHOULD MATCH THIS TEMPLATE:
    
        __generated_sempipes_choices = skrub.choose_from([<Your suggested parameter values>])
        """
