import skrub
from sklearn.base import BaseEstimator
from skrub import selectors
from skrub import DataOp
from skrub._data_ops._skrub_namespace import SkrubNamespace
from sempipes.operators.operators import SemChoices
from sempipes.operators.sem_choose_llm import SemChooseLLM
from sempipes.operators.sem_extract_features import SemExractFeaturesLLM
from sempipes.operators.sem_select_llm import SemSelectLLM
from sempipes.operators.with_sem_features_caafe import WithSemFeaturesCaafe
from sempipes.operators.sem_fillna_llm_plus_model import SemFillNALLLMPlusModel
from sempipes.operators.sem_fillna_llm import SemFillNAWithLLLM
from sempipes.config import set_config, Config, LLM


def sem_choose(name, **kwargs) -> SemChoices:
    return SemChoices(name=name, params_and_prompts=kwargs)


def apply_with_sem_choose(
    self: DataOp,
    estimator: BaseEstimator,
    choices: SemChoices,
    *,
    y=None,
    cols=selectors.all(),
    exclude_cols=None,
    how: str = "auto",
    allow_reject: bool = False,
    unsupervised: bool = False,
):
    SemChooseLLM().set_params_on_estimator(estimator, choices)

    estimator_application = self.apply(
        estimator,
        y=y,
        cols=cols,
        exclude_cols=exclude_cols,
        how=how,
        allow_reject=allow_reject,
        unsupervised=unsupervised,
    )

    choices_storage = skrub.var(f"sempipes__choices__{choices.name}__choices", choices)
    estimator_application = estimator_application.skb.set_name(f"sempipes__choices__{choices.name}__estimator")

    def store_sem_choices(estimator_for_choices, _choices_to_keep):
        # The purpose of this function is just to capture the choice variable in the computational graph
        return estimator_for_choices

    estimator_application = skrub.deferred(store_sem_choices)(estimator_application, choices_storage)

    return estimator_application


def with_sem_features(
    self: DataOp,
    nl_prompt: str,
    name: str,
    how_many: int = 10,
) -> DataOp:
    data_op = self
    feature_gen_estimator = WithSemFeaturesCaafe().generate_features_estimator(data_op, nl_prompt, name, how_many)
    return self.skb.apply(feature_gen_estimator).skb.set_name(name)


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


def sem_select(
    self: DataOp,
    nl_prompt: str,
) -> DataOp:
    selector = SemSelectLLM().generate_column_selector(nl_prompt)
    return self.skb.select(selector)


def sem_extract_features(
    self: DataOp,
    nl_prompt: str,
    input_columns: list[str],
    output_columns: dict[str, str] | None = None,
) -> DataOp:
    feature_extractor = SemExractFeaturesLLM().generate_features_extractor(nl_prompt, input_columns, output_columns)
    return self.skb.apply(feature_extractor)


DataOp.with_sem_features = with_sem_features
DataOp.sem_fillna = sem_fillna
DataOp.sem_select = sem_select
DataOp.sem_extract_features = sem_extract_features
SkrubNamespace.apply_with_sem_choose = apply_with_sem_choose

__all__ = ["sem_choose", "set_config", "Config", "LLM"]
