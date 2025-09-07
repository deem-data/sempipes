from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from gyyre._code_gen._llm import _batch_generate_results
from gyyre._operators import SemFillNAOperator


class SemFillNAWithLLLM(SemFillNAOperator):
    def generate_imputation_estimator(
        self,
        data_op,
        target_column,
        nl_prompt,
        impute_with_existing_values_only=True,
    ):
        return LLMImputer(
            target_column=target_column,
            nl_prompt=nl_prompt,
            impute_with_existing_values_only=impute_with_existing_values_only,
        )


class LLMImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_column: str,
        nl_prompt: str,
        impute_with_existing_values_only: bool,
    ) -> None:
        self.target_column = target_column
        self.nl_prompt = nl_prompt
        self.impute_with_existing_values_only = impute_with_existing_values_only
        self.target_column_type_ = None
        self.target_column_unique_values_ = None

    @staticmethod
    def _build_prompt(
        target_column,
        target_column_type,
        candidate_columns,
        nl_prompt,
        target_column_unique_values,
        impute_existing_values_only,
    ):
        # TODO handle this intelligently
        assert len(target_column_unique_values) < 100, "Too many unique values to include in the prompt."
        target_column_unique_values = ", ".join([str(value) for value in target_column_unique_values])

        if impute_existing_values_only:
            values_instruction = f""" The target column has the following unique values: {target_column_unique_values}. Use only existing values to fill in the missing values. """
        else:
            values_instruction = f" Example values from the {target_column} column: {target_column_unique_values}. "

        return f"""
        The data scientist wants to fill missing values in the column '{target_column}' of type '{target_column_type}' in a dataframe. 
        The dataframe has the following columns available with these values to help with this task: {candidate_columns}. You need to assist the data scientist with filling the missing values in the target column.
        {values_instruction}
        
        The data scientist wants you to take special care to the following: 
        {nl_prompt}.
        
        Answer with only the value to impute, without any additional text or formatting.
        Result:    
        """

    def fit(self, df, y=None):  # pylint: disable=unused-argument
        print(f"--- Sempipes.sem_fillna_llm('{self.target_column}', '{self.nl_prompt}')")

        self.target_column_type_ = df[self.target_column].dtype
        self.target_column_unique_values_ = list(df[self.target_column].dropna().unique())
        # TODO maybe add examples of good imputations

        return self

    def transform(self, df):
        check_is_fitted(self, ["target_column_type_", "target_column_unique_values_"])

        indices_to_impute = df[self.target_column].isna()
        rows_to_impute = df[indices_to_impute]

        if rows_to_impute.shape[0] == 0:
            print("\t> No missing values to impute.")
            return df

        # TODO Should we check if there are more previously unseen unique values in the target column?

        # Build prompts for each row to impute
        prompts = []
        for __, row in rows_to_impute.iterrows():
            candidate_columns = {column: row[column] for column in df.columns if column != self.target_column}
            prompt = self._build_prompt(
                self.target_column,
                self.target_column_type_,
                candidate_columns,
                self.nl_prompt,
                self.target_column_unique_values_,
                self.impute_with_existing_values_only,
            )
            prompts.append(prompt)

        # TODO find a way to set the batch size
        results = _batch_generate_results(prompts, batch_size=10)

        df.loc[indices_to_impute, self.target_column] = results

        print(f"\t> Imputed {len(results)} values...")

        return df
