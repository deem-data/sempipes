import logging
import os

import litellm

os.environ.setdefault("LITELLM_LOGS", "error")
litellm.logger_config = {"success": False, "request": False, "response": False, "cache": False}  # type: ignore[attr-defined]
litellm.set_verbose = False

logging.basicConfig(format="%(asctime)s - %(levelname)s - SEMPIPES> %(message)s", level=logging.INFO)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)
