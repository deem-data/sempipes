from skrub import selectors
from skrub._data_ops._data_ops import DataOp
from skrub._data_ops._skrub_namespace import SkrubNamespace

from gyyre.operator_impls._sem_choose_llm import SemChooseLLM
from gyyre.operator_impls._with_sem_features_caafe import WithSemFeaturesCaafe
from gyyre.operator_impls._sem_fillna_llm_plus_model import SemFillNALLLMPlusModel

from gyyre.optimisers._memory_loop import optimise_semantic_operator

def sem_choose(**kwargs):
    return kwargs

def apply_with_sem_choose(self, estimator, *, y=None, cols=selectors.all(), exclude_cols=None, how="auto",
    allow_reject=False, unsupervised=False, choices=None,):
    data_op = self
    SemChooseLLM().set_params_on_estimator(data_op, estimator, choices, y=y)
    # TODO forward the * args
    return self.apply(estimator,y=y, cols=cols, exclude_cols=exclude_cols, how=how, allow_reject=allow_reject,
                      unsupervised=unsupervised)


def with_sem_features(self, nl_prompt, name, how_many=10):
    data_op = self
    feature_gen_estimator = WithSemFeaturesCaafe().generate_features_estimator(data_op, nl_prompt, name, how_many)
    return self.skb.apply(feature_gen_estimator).skb.set_name(name)


def sem_fillna(self, target_column, nl_prompt):
    data_op = self
    imputation_estimator = SemFillNALLLMPlusModel().generate_imputation_estimator(data_op, target_column, nl_prompt)
    return self.skb.apply(imputation_estimator)


DataOp.with_sem_features = with_sem_features
DataOp.sem_fillna = sem_fillna
SkrubNamespace.apply_with_sem_choose = apply_with_sem_choose

__all__ = ['sem_choose', 'optimise_semantic_operator']
