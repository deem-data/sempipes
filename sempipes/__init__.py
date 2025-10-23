from dotenv import load_dotenv
from skrub import DataOp
import skrub
from skrub._data_ops._skrub_namespace import SkrubNamespace
from sempipes.operators.operators import SemChoices

from sempipes.operators.sem_augment import sem_augment
from sempipes.operators.sem_refine import sem_refine
from sempipes.operators.sem_choose_llm import sem_choose, apply_with_sem_choose
from sempipes.operators.sem_extract_features import sem_extract_features
from sempipes.operators.sem_select_llm import sem_select
from sempipes.operators.with_sem_agg_join_features import with_sem_agg_join_features
from sempipes.operators.with_sem_features_caafe import with_sem_features
from sempipes.operators import sem_fillna
from sempipes.config import get_config, LLM, update_config

# Load environment variables from a .env file if present
load_dotenv()

# Monkey-patch skrub DataOp to include our semantic operators
DataOp.with_sem_features = with_sem_features
DataOp.with_sem_agg_join_features = with_sem_agg_join_features
DataOp.sem_fillna = sem_fillna
DataOp.sem_select = sem_select
DataOp.sem_augment = sem_augment
DataOp.sem_extract_features = sem_extract_features
DataOp.sem_refine = sem_refine
SkrubNamespace.apply_with_sem_choose = apply_with_sem_choose

__all__ = ["sem_choose", "get_config", "update_config", "LLM"]
