"""Runs model-written pandas code in a restricted subprocess.

Layers of protection (honest summary, this is NOT a perfect security sandbox):
1. AST check: no dunder access, no dangerous names, no file/network attributes.
2. Restricted builtins: no open(), eval(), exec(), getattr(), type(), etc.
   Only pandas, numpy, math, re and datetime can be imported.
3. Separate Python process with a timeout, an empty temp working directory and
   a stripped environment (API keys are NOT visible to the model's code).
For untrusted production use, run the whole project inside Docker.
"""

import ast
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

TIMEOUT_SECONDS = 30

BLOCKED_NAMES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "eval", "exec", "compile", "__import__", "builtins", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "type",
    "importlib", "pathlib", "pickle", "ctypes", "breakpoint", "help",
}

BLOCKED_ATTRS = {
    # file and network access through pandas / numpy
    "to_csv", "to_pickle", "to_excel", "to_json", "to_parquet", "to_sql",
    "to_hdf", "to_feather", "to_clipboard", "to_latex", "to_xml", "to_orc",
    "to_stata", "to_html", "to_markdown", "to_gbq",
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "fromregex",
    # ways to reach the os module
    "os", "sys", "io", "compat", "util", "testing", "lib", "ctypeslib",
    "system", "popen", "environ", "builtins", "importlib", "subprocess",
    # introspection helpers that can hand back the os module (found in testing)
    "inspect", "getmembers", "getmodule", "getmodulename", "getsourcefile",
    "functools", "operator", "itertools", "gc", "warnings", "contextlib",
    "ctypes", "multiprocessing", "threading", "signal", "shutil", "tempfile",
    "pathlib", "glob", "pickle", "marshal", "code", "codeop", "runpy", "pkgutil",
}

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re", "datetime", "statistics"}


def extract_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def validate_ast_safety(code: str) -> Tuple[bool, Optional[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, "SyntaxError: " + str(e)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, "SecurityViolation: import of '" + a.name + "' is not allowed."
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return False, "SecurityViolation: import from '" + str(node.module) + "' is not allowed."
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return False, "SecurityViolation: use of '" + node.id + "' is prohibited."
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, "SecurityViolation: dunder attribute access blocked."
            if node.attr in BLOCKED_ATTRS or node.attr.startswith("read_"):
                return False, "SecurityViolation: attribute '" + node.attr + "' is prohibited."
    return True, None


def clean_value(x: Any) -> Any:
    """Normalizes answers into plain int, float or str."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None
        return round(float(x), 4)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        if len(x) == 1:
            val = x.iloc[0]
            if isinstance(val, (pd.Series, pd.DataFrame)):
                val = val.iloc[0]
            return clean_value(val)
        return str(x.values.tolist())[:200]
    if isinstance(x, (list, tuple, set)):
        if len(x) == 1:
            return clean_value(list(x)[0])
        return str(list(x))[:200]
    if isinstance(x, dict):
        return str(x)[:200]
    return str(x).strip()


# This script runs inside the child process.
_RUNNER = r'''
import sys, json, pickle, builtins, math
import pandas as pd
import numpy as np

with open(sys.argv[1], "rb") as f:
    payload = pickle.load(f)
code, df = payload["code"], payload["df"]

ALLOWED = {"pandas", "numpy", "math", "re", "datetime", "statistics"}
_real_import = builtins.__import__

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED:
        raise ImportError("import of " + name + " is not allowed")
    return _real_import(name, globals, locals, fromlist, level)

names = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
         "int", "len", "list", "map", "max", "min", "range", "round", "set",
         "sorted", "str", "sum", "tuple", "zip", "isinstance", "reversed",
         "divmod", "pow", "repr", "print", "frozenset", "slice", "next", "iter",
         "ValueError", "KeyError", "TypeError", "Exception", "ZeroDivisionError",
         "IndexError", "AttributeError", "StopIteration"]
safe = {n: getattr(builtins, n) for n in names}
safe["__import__"] = safe_import

ns = {"__builtins__": safe, "df": df, "pd": pd, "np": np, "math": math}
exec(code, ns)
ans = ns.get("answer")

def conv(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return {"kind": "table", "value": x.values.tolist()}
    if isinstance(x, np.generic):
        return conv(x.item())
    if isinstance(x, (int, float, str, bool)) or x is None:
        return {"kind": "scalar", "value": x}
    if isinstance(x, (list, tuple, set)):
        return {"kind": "list", "value": list(x)}
    return {"kind": "other", "value": str(x)}

with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(conv(ans), f, default=str)
'''


def run_code(code: str, df: pd.DataFrame) -> Tuple[Any, Optional[str]]:
    """Returns (answer, error_string). Never raises."""
    if not code or not code.strip():
        return None, "EmptyCodeError: no code provided."
    ok, sec_err = validate_ast_safety(code)
    if not ok:
        return None, sec_err

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.pkl")
        out_path = os.path.join(tmp, "out.json")
        run_path = os.path.join(tmp, "runner.py")
        with open(in_path, "wb") as f:
            pickle.dump({"code": code, "df": df}, f)
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(_RUNNER)

        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        if os.environ.get("SYSTEMROOT"):                 # needed on Windows
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        try:
            proc = subprocess.run(
                [sys.executable, run_path, in_path, out_path],
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None, "TimeoutError: code ran longer than " + str(TIMEOUT_SECONDS) + " seconds."

        if proc.returncode != 0 or not os.path.exists(out_path):
            lines = [l for l in (proc.stderr or "").strip().splitlines() if l.strip()]
            return None, (lines[-1] if lines else "ExecutionError: unknown failure")[:300]

        with open(out_path, "r", encoding="utf-8") as f:
            res = json.load(f)

    val = res["value"]
    if res["kind"] == "table":
        flat = [v for row in val for v in (row if isinstance(row, list) else [row])]
        return (flat[0] if len(flat) == 1 else str(val)[:200]), None
    if res["kind"] == "list":
        return (val[0] if len(val) == 1 else str(val)[:200]), None
    return val, None


def code_answer_to_dict(ans: Any, err: Optional[str]) -> Dict[str, Any]:
    if err is not None or ans is None:
        return {"cannot_answer": None, "answer": None, "parse_error": True}
    if isinstance(ans, str) and ans.strip().upper() == "CANNOT_ANSWER":
        return {"cannot_answer": True, "answer": None, "parse_error": False}
    return {"cannot_answer": False, "answer": clean_value(ans), "parse_error": False}
