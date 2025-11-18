# pylint: disable=import-error
# These are optional dev dependencies that may not be installed in all environments
import ast
import builtins
import collections
import dataclasses
import datetime
import json
import pathlib
import re
import typing
import unicodedata
import urllib
from collections.abc import Iterable
from typing import Any

import huggingface_hub
import numpy
import open_clip
import pandas as pd
import PIL
import scipy
import sdv
import sklearn
import skrub
import soundfile
import tensorflow
import torch
import torchaudio
import tqdm
import transformers
import unidecode
import xgboost
import autogluon

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
    "torchaudio",
    "soundfile",
    "pathlib",
    "open_clip",
    "urllib",
    "tensorflow",
    "huggingface_hub",
    "unicodedata",
    "unidecode",
    "xgboost",
    "autogluon",
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
        "hasattr": hasattr,
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
        "filter": filter,
        "bin": bin,
        "object": object,
        "Exception": Exception,
        "BaseException": BaseException,
        "ImportError": ImportError,
        "next": next,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "SyntaxError": SyntaxError,
        "enumerate": enumerate,
        "print": print,
        "getattr": getattr,
        "hasattr": hasattr,
        "callable": callable,
        "setattr": setattr,
        "type": type,
        "repr": repr,
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
        "torchaudio": torchaudio,
        "soundfile": soundfile,
        "pathlib": pathlib,
        "open_clip": open_clip,
        "urllib": urllib,
        "tensorflow": tensorflow,
        "huggingface_hub": huggingface_hub,
        "unicodedata": unicodedata,
        "unidecode": unidecode,
        "xgboost": xgboost,
        "autogluon": autogluon,
    }

    # We need a single dict to allow function definitions inside the code
    safe_globals_and_locals = {**safe_globals, **safe_locals_to_add}
    exec(python_code, safe_globals_and_locals, safe_globals_and_locals)

    return safe_globals_and_locals[variable_to_return]
