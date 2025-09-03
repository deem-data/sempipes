# TODO This class needs some serious cleanup
import skrub

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from gyyre._operators import WithSemFeaturesOperator
from gyyre.code_gen._llm import _generate_python_code_from_messages
from gyyre.code_gen._exec import _safe_exec

from gyyre.operator_impls._mixins import GyyrePrefittedMixin, GyyreMemoryMixin

# # Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
# This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".
def _get_prompt(df, nl_prompt, how_many, data_description_unparsed=None, samples=None):
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}

This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting a target label.
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks.

The data scientist wants you to take special care to the following: {nl_prompt}.


Code formatting for each added column:
```python
# (Feature name and description)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Each codeblock generates up to {how_many} useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance.
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""



def _build_prompt_from_df(df, nl_prompt, how_many):
    samples = ""
    df_ = df.head(10)
    for column in list(df_):
        nan_freq = "%s" % float("%.2g" % (df[column].isna().mean() * 100))
        sampled_values = df_[column].tolist()
        if str(df[column].dtype) == "float64":
            sampled_values = [round(sample, 2) for sample in sampled_values]
        samples += (
            f"{df_[column].name} ({df[column].dtype}): NaN-freq [{nan_freq}%], Samples {sampled_values}\n"
        )

    return _get_prompt(
        df,
        nl_prompt,
        how_many,
        data_description_unparsed=None,
        samples=samples,
    )

# TODO this should be moved to the exec code
def strip_code_fences(s: str) -> str:
    # remove lines starting with ``` or ```python (ignoring leading spaces)
    lines = s.splitlines(keepends=True)
    keep = [ln for ln in lines if not ln.lstrip().startswith("```")]
    return "".join(keep)

class LLMFeatureGenerator(BaseEstimator, TransformerMixin, GyyrePrefittedMixin, GyyreMemoryMixin):

    def __init__(self, nl_prompt, how_many, gyyre_prefitted_state=None, gyyre_memory=None):
        self.nl_prompt = nl_prompt
        self.how_many = how_many
        self.gyyre_prefitted_state = gyyre_prefitted_state
        self.gyyre_memory = gyyre_memory

        self.generated_code_ = []

    def state_after_fit(self):
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self):

        if self.generated_code_ is not None and len(self.generated_code_) > 0:
            latest_code = self.generated_code_[-1]
        else:
            latest_code = ""

        return latest_code

    def fit(self, df, y=None, **fit_params):

        prompt_preview = self.nl_prompt[:40].replace("\n", " ").strip()

        if self.gyyre_prefitted_state is not None:
            print(f"--- Skipping fit, using provided state for gyyre.with_sem_features('{prompt_preview}...', {self.how_many})")
            self.generated_code_ = self.gyyre_prefitted_state["generated_code"]
        else:
            print(f"--- Fitting gyyre.with_sem_features('{prompt_preview}...', {self.how_many})")
            max_retries = 5
            messages = []
            for attempt in range(1, max_retries + 1):

                code = ""

                try:
                    prompt = _build_prompt_from_df(df, self.nl_prompt, self.how_many)

                    if attempt == 1:
                        messages += [{
                            "role": "system", "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",},
                            {"role": "user", "content": prompt,},
                        ]

                        if self.gyyre_memory is not None and len(self.gyyre_memory) > 0:

                            current_accuracy = 0.0
                            current_roc = 0.0

                            for memory_line in self.gyyre_memory:

                                memorized_code = memory_line["update"]
                                memorized_accuracy = memory_line["accuracy"]
                                # TODO also compute and provide ROC AUC
                                memorized_roc = memory_line["accuracy"]

                                improvement_acc = memorized_accuracy - current_accuracy
                                improvement_roc = memorized_roc - current_roc

                                if improvement_roc + improvement_acc >= 0.0:
                                    self.generated_code_.append(memorized_code)
                                    add_feature_sentence = "The code was executed and changes to ´df´ were kept."

                                    current_accuracy = memorized_accuracy
                                    current_roc = memorized_roc

                                else:
                                    add_feature_sentence = f"The last code changes to ´df´ were discarded. (Improvement: {improvement_roc + improvement_acc})"

                                messages += [
                                    {"role": "assistant", "content": memorized_code},
                                    {
                                        "role": "user",
                                        "content": f"""Performance after adding feature ROC {memorized_roc:.3f}, ACC {memorized_accuracy:.3f}. {add_feature_sentence}
                            Next codeblock:
                            """,
                                    },
                                ]

                    code = _generate_python_code_from_messages(messages)
                    code = strip_code_fences(code)

                    code_to_execute = "\n".join(self.generated_code_)
                    code_to_execute += "\n\n" + code

                    df_sample = df.head(100).copy(deep=True)
                    columns_before = df_sample.columns
                    df_sample_processed = _safe_exec(code_to_execute, "df", safe_locals_to_add={"df": df_sample})
                    columns_after = df_sample_processed.columns
                    new_columns = sorted(set(columns_after) - set(columns_before))
                    removed_columns = sorted(set(columns_before) - set(columns_after))

                    print(f"\t> Computed {len(new_columns)} new feature columns: {new_columns}, removed {len(removed_columns)} feature columns: {removed_columns}")

                    self.generated_code_.append(code)
                    break
                except Exception as e:
                    print(f"\t> An error occurred in attempt {attempt}:", e)
                    messages += [
                        {"role": "assistant", "content": code},
                        {"role": "user",
                            "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                                        ```python
                                        """,
                        },
                    ]

        return self

    def transform(self, df):
        check_is_fitted(self, "generated_code_")
        code_to_execute = "\n".join(self.generated_code_)
        df = _safe_exec(code_to_execute, "df", safe_locals_to_add={"df": df})
        return df

class WithSemFeaturesCaafe(WithSemFeaturesOperator):
    def generate_features_estimator(self, data_op, nl_prompt, name, how_many):
        # TODO explore computational graph or cached preview results to improve feature generation

        gyyre_prefitted_state = skrub.var(f"gyyre_prefitted_state__{name}", None)
        gyyre_memory = skrub.var(f"gyyre_memory__{name}", [])

        return LLMFeatureGenerator(nl_prompt, how_many,
            gyyre_prefitted_state=gyyre_prefitted_state, gyyre_memory=gyyre_memory)