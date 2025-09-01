from sempipes._features import with_sem_features
from sempipes._fill_na import sem_fillna
from sempipes._choose import sem_choose, apply_with_sem_choose

from skrub._data_ops._data_ops import DataOp
from skrub._data_ops._skrub_namespace import SkrubNamespace

DataOp.with_sem_features = with_sem_features
DataOp.sem_fillna = sem_fillna
SkrubNamespace.apply_with_sem_choose = apply_with_sem_choose

__all__ = ['sem_choose']