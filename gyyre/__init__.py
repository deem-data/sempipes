from sklearn.base import BaseEstimator
from skrub import selectors
from skrub import DataOp
from skrub._data_ops._skrub_namespace import SkrubNamespace

from gyyre._operator_impls._sem_choose_llm import SemChooseLLM
from gyyre._operator_impls._sem_select_llm import SemSelectLLM
from gyyre._operator_impls._with_sem_features_caafe import WithSemFeaturesCaafe
from gyyre._operator_impls._sem_fillna_llm_plus_model import SemFillNALLLMPlusModel
from gyyre._operator_impls._sem_fillna_llm import SemFillNAWithLLLM
from gyyre.optimisers.greedy import greedy_optimise_semantic_operator


def sem_choose(**kwargs) -> dict:
    return kwargs


def apply_with_sem_choose(
    self: DataOp,
    estimator: BaseEstimator,
    *,
    y=None,
    cols=selectors.all(),
    exclude_cols=None,
    how: str = "auto",
    allow_reject: bool = False,
    unsupervised: bool = False,
    choices=None,
):
    data_op = self
    SemChooseLLM().set_params_on_estimator(data_op, estimator, choices, y=y)
    # TODO forward the * args
    return self.apply(
        estimator,
        y=y,
        cols=cols,
        exclude_cols=exclude_cols,
        how=how,
        allow_reject=allow_reject,
        unsupervised=unsupervised,
    )


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


DataOp.with_sem_features = with_sem_features
DataOp.sem_fillna = sem_fillna
DataOp.sem_select = sem_select
SkrubNamespace.apply_with_sem_choose = apply_with_sem_choose

__all__ = ["sem_choose", "greedy_optimise_semantic_operator"]
