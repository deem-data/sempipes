from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sempipes._operators import WithSemFeaturesOperator
from sempipes.code_gen._llm import _generate_python_code
from sempipes.code_gen._exec import _safe_exec


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
# --COLUMN--START
# Feature name
# (Feature name and description)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
# --COLUMN--END
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

class LLMFeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, nl_prompt, how_many):
        self.nl_prompt = nl_prompt
        self.how_many = how_many
        self.generated_code_ = None

    def fit(self, df, y=None, **fit_params):
        print(f"--- Sempipes.with_sem_features('{self.nl_prompt}', {self.how_many})")

        prompt = _build_prompt_from_df(df, self.nl_prompt, self.how_many)
        self.generated_code_ = _generate_python_code(prompt)
        #TODO: check if code runs / compiles, retry otherwise
        return self

    def transform(self, df):
        check_is_fitted(self, "generated_code_")

        print(self.generated_code_)
        columns_before = df.columns
        df = _safe_exec(self.generated_code_, "df", safe_locals_to_add={"df": df})
        columns_after = df.columns
        new_columns = sorted(set(columns_after) - set(columns_before))
        print(f"\tGenerated {len(new_columns)} feature columns: {new_columns}")
        return df

class WithSemFeaturesCaafe(WithSemFeaturesOperator):
    def generate_features_estimator(self, data_op, nl_prompt, how_many):
        # TODO explore computational graph or cached preview results to improve feature generation
        return LLMFeatureGenerator(nl_prompt, how_many)