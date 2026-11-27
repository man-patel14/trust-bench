"""
Scoring and Evaluation Engine for Trust Bench.
Implements the exact scoring taxonomy, numeric tolerance matching (0.5%),
refusal validation, and summary metric calculations.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from trustbench.config import QUESTIONS_PATH, results_paths


def norm_text(x: Any) -> str:
    """Normalizes string answers for robust lexical comparison."""
    if x is None:
        return ""
    return str(x).strip().strip(".'\"").lower()


def score_row(r: pd.Series) -> str:
    """
    Classifies an individual model response into the standard Trust Bench outcome taxonomy:
    - correct: Accurate answer within 0.5% tolerance
    - wrong: Incorrect answer to an answerable question
    - over_abstain: Refused to answer when data was available
    - correct_abstain: Correctly refused an unanswerable or ambiguous trap question
    - hallucinated: Fabricated an answer to an unanswerable or ambiguous trap question
    - error: Execution or JSON parsing failure
    """
    if bool(r.get("parse_error", False)):
        return "error"

    kind = str(r.get("kind", ""))
    cannot_answer = bool(r.get("cannot_answer", False))
    _a = r.get("answer")
    if isinstance(_a, str) and _a.strip().strip("\"'").upper() == "CANNOT_ANSWER":
        cannot_answer = True          # the model wrote the refusal as the answer text

    if kind == "abstain":
        return "correct_abstain" if cannot_answer else "hallucinated"

    if cannot_answer:
        return "over_abstain"

    ans, exp = r.get("answer"), r.get("expected")
    if ans is None or pd.isna(ans) or str(ans).strip().lower() in ["none", "null", ""]:
        return "wrong"

    if kind == "number":
        try:
            a = float(str(ans).replace(",", "").replace("$", "").replace("£", "").replace("%", "").strip())
            e = float(str(exp).replace(",", "").strip())
            # 0.5% relative error or 0.01 absolute tolerance
            tol = max(0.01, 0.005 * abs(e))
            return "correct" if abs(a - e) <= tol else "wrong"
        except Exception:
            return "wrong"

    # Text comparison
    return "correct" if norm_text(ans) == norm_text(exp) else "wrong"


def score_results(
    mode: str = "real",
    questions_path: str = QUESTIONS_PATH,
    raw_results_path: Optional[str] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Loads raw jsonl results, joins ground truth, computes outcomes and saves scored.csv."""
    paths = results_paths(mode)
    raw_results_path = raw_results_path or paths["raw"]
    save_path = save_path or paths["scored"]
    if not os.path.exists(raw_results_path):
        raise FileNotFoundError(f"Results file not found: {raw_results_path}")

    rows = []
    with open(raw_results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    res = pd.DataFrame(rows)

    with open(questions_path, "r", encoding="utf-8") as f:
        q_list = json.load(f)

    qs = pd.DataFrame(q_list).rename(columns={"id": "qid"})[
        ["qid", "category", "difficulty", "kind", "expected", "question"]
    ]

    if "simulated" not in res.columns:
        res["simulated"] = False
    res = res.merge(qs, on="qid", how="left")
    res["outcome"] = res.apply(score_row, axis=1)
    res["is_correct"] = res["outcome"].isin(["correct", "correct_abstain"])
    res["answer"] = res["answer"].astype(str)
    res["expected"] = res["expected"].astype(str)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    res.to_csv(save_path, index=False, encoding="utf-8")
    print(f"[TrustBench] Scored {len(res)} predictions saved to {save_path}")
    print(res["outcome"].value_counts())
    return res


def compute_metrics_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Computes comprehensive executive benchmark table across models and approaches."""
    res = scored_df.copy()
    answerable = res[res["kind"] != "abstain"]
    traps = res[res["kind"] == "abstain"]
    g = ["model", "approach"]

    summary = pd.DataFrame({
        "overall_correct": res.groupby(g)["is_correct"].mean(),
        "accuracy_answerable": answerable.groupby(g)["is_correct"].mean(),
        "wrong_answer_rate": answerable.groupby(g)["outcome"].apply(lambda s: (s == "wrong").mean()),
        "over_abstain_rate": answerable.groupby(g)["outcome"].apply(lambda s: (s == "over_abstain").mean()),
        "trap_handled": traps.groupby(g)["is_correct"].mean(),
        "hallucination_rate": traps.groupby(g)["outcome"].apply(lambda s: (s == "hallucinated").mean()),
        "error_rate": res.groupby(g)["outcome"].apply(lambda s: (s == "error").mean()),
        "avg_latency_s": res.groupby(g)["latency"].mean(),
        "total_evaluations": res.groupby(g)["is_correct"].size(),
    }).round(3)

    return summary

