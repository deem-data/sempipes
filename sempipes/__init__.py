from dotenv import load_dotenv
from skrub import DataOp
import skrub
from skrub._data_ops._skrub_namespace import SkrubNamespace
from sempipes.operators.operators import SemChoices

from sempipes.operators.sem_augment import sem_augment
from sempipes.operators.sem_refine import sem_refine
from sempipes.operators.sem_choose_llm import sem_choose, apply_with_sem_choose
from sempipes.operators.sem_distill import sem_distill
from sempipes.operators.sem_extract_features import sem_extract_features
from sempipes.operators.sem_select_llm import sem_select
from sempipes.operators.sem_agg_features import sem_agg_features
from sempipes.operators.sem_gen_features_caafe import sem_gen_features
from sempipes.operators import sem_fillna
from sempipes.config import get_config, LLM, update_config

# Load environment variables from a .env file if present
load_dotenv()

# Monkey-patch skrub DataOp to include our semantic operators
DataOp.sem_fillna = sem_fillna
DataOp.sem_select = sem_select
DataOp.sem_augment = sem_augment
DataOp.sem_distill = sem_distill
DataOp.sem_gen_features = sem_gen_features
DataOp.sem_agg_features = sem_agg_features
DataOp.sem_extract_features = sem_extract_features
DataOp.sem_refine = sem_refine
SkrubNamespace.apply_with_sem_choose = apply_with_sem_choose


def as_X(op: DataOp, description: str):
    """Mark the DataOp as feature set X with a description."""
    return op.skb.mark_as_X().skb.set_description(description)


def as_y(op: DataOp, description: str):
    """Mark the DataOp as label set y with a description."""
    return op.skb.mark_as_y().skb.set_description(description)


__all__ = ["as_X", "as_y", "sem_choose", "get_config", "update_config", "LLM"]
