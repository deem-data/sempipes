import ast
import builtins
import collections
import dataclasses
import datetime
import json
import re
import typing
from collections.abc import Iterable
from typing import Any

import numpy
import pandas as pd
import PIL
import scipy
import sdv
import sklearn
import skrub
import torch
import tqdm
import transformers

_ALLOWED_MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "skrub",
    "re",
    "json",
    "ast",
    "datetime",
    "collections",
    "math",
    "transformers",
    "torch",
    "json",
    "dataclasses",
    "typing",
    "PIL",
    "random",
    "time",
    "tqdm",
    "scipy",
    "sdv",
]


def _make_safe_import(allowed_modules: Iterable[str]):
    real_import = builtins.__import__

    def safe_import(name, globals_to_import=None, locals_to_import=None, fromlist=(), level=0):
        top_name = name.split(".")[0]
        if top_name not in allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed")
        return real_import(name, globals_to_import, locals_to_import, fromlist, level)

    return safe_import


def safe_exec(
    python_code: str,
    variable_to_return: str,
    safe_locals_to_add: dict[str, Any] | None = None,
) -> Any:
    if safe_locals_to_add is None:
        safe_locals_to_add = {}

    safe_builtins = {
        "__import__": _make_safe_import(_ALLOWED_MODULES),
        "__build_class__": builtins.__build_class__,
        "__name__": "__main__",
        "super": super,
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
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "zip": zip,
        "sorted": sorted,
        "object": object,
        "Exception": Exception,
        "BaseException": BaseException,
        "next": next,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "SyntaxError": SyntaxError,
        "enumerate": enumerate,
        "print": print,
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
        "ast": ast,
        "datetime": datetime,
        "collections": collections,
        "transformers": transformers,
        "torch": torch,
        "json": json,
        "dataclasses": dataclasses,
        "typing": typing,
        "PIL": PIL,
        "tqdm": tqdm,
        "scipy": scipy,
        "sdv": sdv,
    }

    # We need a single dict to allow function definitions inside the code
    safe_globals_and_locals = {**safe_globals, **safe_locals_to_add}
    exec(python_code, safe_globals_and_locals, safe_globals_and_locals)

    return safe_globals_and_locals[variable_to_return]
