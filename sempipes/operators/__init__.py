from skrub import DataOp
from sempipes.operators.sem_fillna_llm import SemFillNAWithLLLM
from sempipes.operators.sem_fillna_llm_plus_model import SemFillNALLLMPlusModel


def sem_fillna(
    self: DataOp,
    target_column: str,
    nl_prompt: str,
    impute_with_existing_values_only: bool,
    **kwargs,
) -> DataOp:
    data_op = self

    if "with_llm_only" in kwargs and kwargs["with_llm_only"]:
        imputation_estimator = SemFillNAWithLLLM().generate_imputation_estimator(
            data_op, target_column, nl_prompt, impute_with_existing_values_only
        )
    else:
        # TODO Handle this case better for users
        assert impute_with_existing_values_only
        imputation_estimator = SemFillNALLLMPlusModel().generate_imputation_estimator(data_op, target_column, nl_prompt)
    return self.skb.apply(imputation_estimator)
