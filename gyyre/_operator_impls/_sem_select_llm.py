from typing import Any

from skrub import _dataframe as sbd
from skrub.selectors._base import Filter

from gyyre._code_gen._exec import _safe_exec
from gyyre._code_gen._llm import _generate_python_code
from gyyre._operators import SemSelectOperator


def _sem_select(column: Any, nl_prompt: str) -> bool:
    column_name = sbd.name(column)

    try:
        size = sbd.shape(column)[0]
        sample_size = min(10, size)
        sample = list(sbd.sample(column, sample_size))

        numeric_profile = ""
        if sbd.is_numeric(column):
            numeric_profile = f"""
        ### Numerical information    
        - minimum: {sbd.min(column)}
        - maximum: {sbd.max(column)}
        - mean: {sbd.mean(column)}            
            """

        prompt = f"""

        Your goal is to help a data scientist select relevant columns from a pandas dataframe for a machine learning script 
        which they are developing. You make your decision for each column individually. 

        The data scientist wants you to take special care to the following: 
        {nl_prompt}

        Here is detailed information about a column for which you need to make a decision now:

        ### Details about the column
        - name: {column_name}
        - sample_of_ten_or_less_randomly_chosen_values: {sample}
        - num_unique_values: {sbd.n_unique(column)}
        - has_nulls ? {sbd.has_nulls(column)}
        - is_all_null ? {sbd.is_all_null(column)}
        - is_sorted ? {sbd.is_sorted(column)}

        ### Type information about the column
        - is_bool ? {sbd.is_bool(column)}
        - is_numeric ? {sbd.is_numeric(column)}
        - is_integer ? {sbd.is_integer(column)}
        - is_float ? {sbd.is_float(column)}
        - is_string ? {sbd.is_string(column)}
        - is_object ? {sbd.is_object(column)}
        - is_any_date ? {sbd.is_any_date(column)}
        - is_duration ? {sbd.is_duration(column)}
        - is_categorical ? {sbd.is_categorical(column)}

        {numeric_profile}

        Please respond with a piece of Python code that expresses your decision as a boolean. 
        ONLY RESPOND WITH THE CODE BLOCK, NOTHING ELSE.

        ```python
        __column_is_relevant = <True or False>
        ```            
        """

        python_code = _generate_python_code(prompt)
        decision = _safe_exec(python_code, "__column_is_relevant")

        return decision

    except Exception as e:  # pylint: disable=broad-except
        print(f"An error occurred for column {column_name}:", e)
        return False


class SemSelectLLM(SemSelectOperator):
    def generate_column_selector(
        self,
        nl_prompt: str,
    ) -> Filter:
        return Filter(_sem_select, args=(nl_prompt,), name="sem_select")
