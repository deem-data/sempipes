import builtins
import re
from collections.abc import Iterable
from typing import Any

import numpy
import pandas as pd
import sklearn
import skrub

_ALLOWED_MODULES = ["numpy", "pandas", "sklearn", "skrub", "re"]


def _make_safe_import(allowed_modules: Iterable[str]):
    real_import = builtins.__import__

    def safe_import(name, globals_to_import=None, locals_to_import=None, fromlist=(), level=0):
        top_name = name.split(".")[0]
        if top_name not in allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed")
        return real_import(name, globals_to_import, locals_to_import, fromlist, level)

    return safe_import


def _safe_exec(
    python_code: str,
    variable_to_return: str,
    safe_locals_to_add: dict[str, Any] | None = None,
) -> Any:
    if safe_locals_to_add is None:
        safe_locals_to_add = {}

    safe_builtins = {
        "__import__": _make_safe_import(_ALLOWED_MODULES),
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "len": len,
        "range": range,
        "isinstance": isinstance,
        "sum": sum,
        "any": any,
        "all": all,
        "map": map,
        "hash": hash,
    }

    safe_globals = {
        "__builtins__": safe_builtins,
        "skrub": skrub,
        "sklearn": sklearn,
        "numpy": numpy,
        "np": numpy,
        "pandas": pd,
        "pd": pd,
        "re": re,
    }

    safe_locals = safe_locals_to_add
    exec(python_code, safe_globals, safe_locals_to_add)

    return safe_locals[variable_to_return]
