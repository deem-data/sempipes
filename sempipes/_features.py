from sempipes._inspection import context_graph
from sempipes.code_gen._llm import _generate_python_code
from sempipes.code_gen._exec import _safe_exec

# # Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
# This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".

#TODO Taken from CAAFE, give them credit
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

def _internal_with_sem_features(data_op, nl_prompt, how_many):
    ctx = context_graph(data_op._skrub_impl)
    def sempipes_with_sem_features(df):
        print(f"--- Sempipes.with_sem_features('{nl_prompt}', {how_many})")

        prompt = _build_prompt_from_df(df, nl_prompt, how_many)
        python_code = _generate_python_code(prompt)

        #print(python_code)
        columns_before = df.columns
        df = _safe_exec(python_code, "df", safe_locals_to_add={"df": df})
        columns_after = df.columns
        new_columns = sorted(set(columns_after) - set(columns_before))
        print(f"\tGenerated {len(new_columns)} feature columns: {new_columns}")
        return df
    return sempipes_with_sem_features

def with_sem_features(self, nl_prompt, how_many=10):
    return self.skb.apply_func(_internal_with_sem_features(self, nl_prompt, how_many))