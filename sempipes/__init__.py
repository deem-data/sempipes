import os

# Set SCIPY_ARRAY_API=1 to enable scipy array API compatibility. Must be set before any scipy imports.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
from dotenv import load_dotenv  # pylint: disable=wrong-import-position
from skrub import DataOp  # pylint: disable=wrong-import-position
from skrub._data_ops._skrub_namespace import SkrubNamespace  # pylint: disable=wrong-import-position
from sempipes.operators.sem_augment import sem_augment  # pylint: disable=wrong-import-position
from sempipes.operators.sem_refine import sem_refine  # pylint: disable=wrong-import-position
from sempipes.operators.sem_choose_llm import sem_choose, apply_with_sem_choose  # pylint: disable=wrong-import-position
from sempipes.operators.sem_distill import sem_distill  # pylint: disable=wrong-import-position
from sempipes.operators.sem_extract_features import sem_extract_features  # pylint: disable=wrong-import-position
from sempipes.operators.sem_select_llm import sem_select  # pylint: disable=wrong-import-position
from sempipes.operators.sem_agg_features import sem_agg_features  # pylint: disable=wrong-import-position
from sempipes.operators.sem_gen_features_caafe import sem_gen_features  # pylint: disable=wrong-import-position
from sempipes.operators.sem_clean import sem_clean  # pylint: disable=wrong-import-position
from sempipes.operators import sem_fillna  # pylint: disable=wrong-import-position
from sempipes.config import get_config, LLM, update_config, detect_interactive_mode  # pylint: disable=wrong-import-position
from sempipes.logging import get_logger  # pylint: disable=wrong-import-position
from sempipes.interactive.inspection import inspect_generated_code  # pylint: disable=wrong-import-position

# Load environment variables from a .env file if present
load_dotenv()

# Detect if code is running in a Jupyter notebook, to determine the behavior of the preview mode.
if detect_interactive_mode():
    get_logger().info("Interactive mode detected, enabling code generation in preview mode.")
    update_config(prefer_empty_state_in_preview=False)


# Monkey-patch skrub DataOp to include our semantic operators
DataOp.sem_fillna = sem_fillna
DataOp.sem_select = sem_select
DataOp.sem_augment = sem_augment
DataOp.sem_distill = sem_distill
DataOp.sem_gen_features = sem_gen_features
DataOp.sem_agg_features = sem_agg_features
DataOp.sem_extract_features = sem_extract_features
DataOp.sem_refine = sem_refine
DataOp.sem_clean = sem_clean
SkrubNamespace.apply_with_sem_choose = apply_with_sem_choose


def as_X(op: DataOp, description: str):
    """Mark the DataOp as feature set X with a description."""
    return op.skb.mark_as_X().skb.set_description(description)


def as_y(op: DataOp, description: str):
    """Mark the DataOp as label set y with a description."""
    return op.skb.mark_as_y().skb.set_description(description)


__all__ = ["as_X", "as_y", "sem_choose", "get_config", "update_config", "LLM", "inspect_generated_code"]
