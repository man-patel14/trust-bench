"""
Root Helper Module for Trust Bench.
Provides identical API compatibility for Jupyter Notebooks 01 through 04 as documented
in the Trust Bench guide.
"""

import os
import pandas as pd
from typing import Dict, Any, List

from trustbench.config import (
    TABLE_PATH,
    RESULTS_PATH,
    QUESTIONS_PATH,
    SCORED_PATH,
    results_paths,
    MODELS,
    PRICES,
    prompt_direct,
    prompt_code,
    prompt_check,
)
from trustbench.executor import (
    extract_code,
    run_code,
    clean_value,
    code_answer_to_dict,
)
from trustbench.models import (
    call_llm,
    parse_json_answer,
    solve_all,
)


def load_table(path: str = TABLE_PATH) -> pd.DataFrame:
    """Loads benchmark table with appropriate dtypes and datetime parsing."""
    return pd.read_csv(
        path,
        parse_dates=["order_datetime"],
        dtype={"order_id": str, "product_code": str, "customer_id": "Int64"},
    )


__all__ = [
    "TABLE_PATH",
    "RESULTS_PATH",
    "QUESTIONS_PATH",
    "SCORED_PATH",
    "results_paths",
    "MODELS",
    "PRICES",
    "load_table",
    "call_llm",
    "solve_all",
    "prompt_direct",
    "prompt_code",
    "prompt_check",
    "parse_json_answer",
    "extract_code",
    "run_code",
    "clean_value",
    "code_answer_to_dict",
]

