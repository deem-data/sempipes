import pandas as pd
import skrub
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.operators.operators import EstimatorTransformer, WithSemAggJoinFeaturesOperator


def _dataframe_mini_summary(df: pd.DataFrame, sample_size: int = 10) -> str:
    summary_lines = []

    for column in df.columns:
        column_type = df[column].dtype
        missing_ratio = df[column].isna().mean()
        sample_values = (
            df[column].dropna().sample(n=min(sample_size, df[column].dropna().shape[0]), random_state=42).tolist()
            if df[column].notna().any()
            else []
        )

        summary_lines.append(
            f"Column: {column}\n"
            f"  Type: {column_type}\n"
            f"  Missing ratio: {missing_ratio:.2%}\n"
            f"  Sample values: {sample_values}\n"
        )

    return "\n".join(summary_lines)


def _stack_with_source(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    stacked = pd.concat({"left": left, "right": right}, names=["source"], sort=False)
    stacked.attrs["restore"] = {"left": list(left.columns), "right": list(right.columns)}
    return stacked


def _restore_original(stacked: pd.DataFrame, source_key: str) -> pd.DataFrame:
    if "source" not in stacked.index.names:
        raise ValueError("Expected a MultiIndex with level named 'source'.")

    meta = stacked.attrs.get("restore", {})
    if source_key not in meta:
        raise ValueError(f"Missing restoration metadata for key: {source_key}")

    cols = meta[source_key]
    restored = stacked.xs(source_key, level="source").reindex(columns=cols)
    return restored


_SYSTEM_PROMPT = "You are an expert data scientist assistant helping data scientists write a data preprocessing pipeline for a predictive model. "

_MAX_RETRIES = 5


def _build_prompt(left_df, right_df, left_join_column, right_join_column, nl_prompt, how_many):
    left_df_summary = _dataframe_mini_summary(left_df)
    right_df_summary = _dataframe_mini_summary(right_df)

    # TODO Add few-shot examples
    return f"""
        You need to help the data scientist with generating additional features for their training data. They already have a dataframe with 
        existing features and want to left join another dataframe with the existing data frame to generate more features. For that, they need
        to decide which columns to include from the dataframe to join and how to aggregate them in a meaningful way. A single column can be 
        included multiple times with different aggregations.

        The dataframe with the existing training data looks as follows:

        {left_df_summary}

        The `left_join_column` is: {left_join_column}

        The dataframe to left join and aggregate looks as follows:

        {right_df_summary}

        The `right_join_column` is: {right_join_column}

        Here is the full output of df.describe(include='all') for the dataframe to join. You can use these statistics
        as constants for your aggregation functions if needed.
        
        {right_df.describe(include='all').to_string()}

        The data scientist wants you to take special care of the following:

        {nl_prompt}

        Generate a Python function called `_sem_agg_join` that takes four arguments: `left_join_column`, `left_df`, `right_join_column` and `right_df` 
        and conducts the desired left join and aggregations. Your code should generate {how_many} new features.
        
        DO NOT INCLUDE EXAMPLE USAGE CODE. WRAP YOUR RESPONSE CODE IN ```python and ```.

        MAKE SURE THAT THE NEW COLUMNS HAVE MEANINGFUL NAMES.
        
        EXPLAIN YOUR RATIONALE FOR CHOOSING AGGREGATION FUNCTIONS IN COMMENTS IN THE PYTHON CODE.     
    """


def _try_to_execute(generated_code, left_df, left_join_key, right_df, right_join_key):
    agg_join_func = safe_exec(generated_code, variable_to_return="_sem_agg_join")

    # print("#" * 80)
    # print(generated_code)
    # print("#" * 80)

    left_sample = left_df.head(n=100)
    left_keys = left_sample[left_join_key].sample(frac=0.9, random_state=42)
    right_sample = right_df[right_df[right_join_key].isin(left_keys)]
    test_result = agg_join_func(left_join_key, left_sample, right_join_key, right_sample)

    if right_join_key in test_result.columns:
        test_result = test_result.drop(columns=[right_join_key])

    assert isinstance(test_result, pd.DataFrame)
    assert test_result.shape[0] == left_sample.shape[0]

    assert set(left_sample.columns).issubset(
        set(test_result.columns)
    ), "Not all columns from the left input are retained"

    return test_result


# TODO Should be context-aware, optimisable, prefittable
class LLMCodeGenSemAggJoinFeaturesEstimator(EstimatorTransformer):
    def __init__(self, left_join_key: str, right_join_key: str, nl_prompt: str, how_many: int):
        self.left_join_key = left_join_key
        self.right_join_key = right_join_key
        self.nl_prompt = nl_prompt
        self.how_many = how_many
        self.generated_code_: str | None = None

    def fit(self, stacked_inputs, y=None) -> "LLMCodeGenSemAggJoinFeaturesEstimator":  # pylint: disable=unused-argument
        samples = _restore_original(stacked_inputs, source_key="left")
        data_to_aggregate = _restore_original(stacked_inputs, source_key="right")

        prompt = _build_prompt(
            samples,
            data_to_aggregate,
            self.left_join_key,
            self.right_join_key,
            self.nl_prompt,
            self.how_many,
        )

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        for attempt in range(1, _MAX_RETRIES + 1):
            code = generate_python_code_from_messages(messages)
            try:
                test_result = _try_to_execute(code, samples, self.left_join_key, data_to_aggregate, self.right_join_key)
                new_columns = [column for column in test_result.columns if column not in samples.columns]

                print(f"\t> Computed {len(new_columns)} new feature columns: {new_columns}.")
                self.generated_code_ = code
                break

            except Exception as e:  # pylint: disable=broad-except
                print(f"\t> An error occurred in attempt {attempt}:", e)

                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed with error: {type(e)} {e}.\n "
                        + f"Code: ```python{code}```\n Retry and fix the errors!\n```python\n",
                    },
                ]

        return self

    def transform(self, stacked_inputs) -> pd.DataFrame:
        check_is_fitted(self, "generated_code_")
        samples = _restore_original(stacked_inputs, source_key="left")
        data_to_aggregate = _restore_original(stacked_inputs, source_key="right")

        num_samples_before = len(samples)

        agg_join_func = safe_exec(self.generated_code_, variable_to_return="_sem_agg_join")  # type: ignore
        result_df = agg_join_func(self.left_join_key, samples, self.right_join_key, data_to_aggregate)

        if self.right_join_key in result_df.columns:
            result_df = result_df.drop(columns=[self.right_join_key])

        num_samples_after = len(result_df)
        assert num_samples_before == num_samples_after

        return result_df


def _prepare_unstack(df):
    return len(df), df.index


def _unstack(df, n_and_original_index):
    n, original_index = n_and_original_index
    result_df = df.head(n)
    result_df.index = original_index
    return result_df


def with_sem_agg_join_features(  # pylint: disable=too-many-positional-arguments
    self: DataOp,
    right_data_op: DataOp,
    left_on: str,
    right_on: str,
    nl_prompt: str,
    name: str,
    how_many: int = 10,
) -> DataOp:
    left_data_op = self
    # This is a hack to get around the fact that skrub does not support multi-input estimators
    stacked = skrub.deferred(_stack_with_source)(left_data_op, right_data_op).skb.set_name(f"stacked_{name}_inputs")
    len_and_index_left_df = skrub.deferred(_prepare_unstack)(left_data_op).skb.set_name(f"prepare_unstack_{name}")

    agg_joiner = LLMCodeGenSemAggJoinFeaturesOperator().generate_agg_join_features_estimator(
        left_join_key=left_on,
        right_join_key=right_on,
        nl_prompt=nl_prompt,
        how_many=how_many,
    )

    result = stacked.skb.apply(agg_joiner).skb.set_name(name)
    # This is required as the stacking is also applied at transform time, so we need to undo it
    unstacked_result = skrub.deferred(_unstack)(result, len_and_index_left_df).skb.set_name(f"unstacked_{name}")
    return unstacked_result


class LLMCodeGenSemAggJoinFeaturesOperator(WithSemAggJoinFeaturesOperator):
    def generate_agg_join_features_estimator(
        self, left_join_key: str, right_join_key: str, nl_prompt: str, how_many: int
    ) -> EstimatorTransformer:
        return LLMCodeGenSemAggJoinFeaturesEstimator(
            left_join_key=left_join_key,
            right_join_key=right_join_key,
            nl_prompt=nl_prompt,
            how_many=how_many,
        )
