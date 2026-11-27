"""Statistics, cost and charts for Trust Bench."""

import os
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustbench.config import PRICES


def bootstrap_ci(values, n: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_question_correct(res: pd.DataFrame, model: str, approach: str) -> pd.Series:
    g = res[(res["model"] == model) & (res["approach"] == approach)]
    return g.groupby("qid")["is_correct"].mean()


def compute_confidence_intervals(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, approach), grp in res.groupby(["model", "approach"]):
        per_q = grp.groupby("qid")["is_correct"].mean()
        lo, hi = bootstrap_ci(per_q.values)
        rows.append({"model": model, "approach": approach, "mean": round(float(per_q.mean()), 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3), "n_questions": len(per_q)})
    return pd.DataFrame(rows)


def paired_difference(res: pd.DataFrame, a: Tuple[str, str], b: Tuple[str, str],
                      n: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Paired bootstrap on the SAME questions: is A really better than B?
    a and b are (model, approach). If the interval contains 0, there is no clear winner."""
    pa, pb = per_question_correct(res, *a), per_question_correct(res, *b)
    common = pa.index.intersection(pb.index)
    diff = (pa[common] - pb[common]).values
    lo, hi = bootstrap_ci(diff, n=n, seed=seed)
    return {"a": a, "b": b, "mean_diff": round(float(diff.mean()), 3),
            "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "clear_winner": bool(lo > 0 or hi < 0), "n_questions": len(common)}


def all_paired_differences(res: pd.DataFrame) -> pd.DataFrame:
    combos = sorted(res[["model", "approach"]].drop_duplicates().itertuples(index=False, name=None))
    rows = []
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            d = paired_difference(res, combos[i], combos[j])
            rows.append({"A": "%s | %s" % d["a"], "B": "%s | %s" % d["b"],
                         "A_minus_B": d["mean_diff"], "ci_low": d["ci_low"],
                         "ci_high": d["ci_high"], "clear_winner": d["clear_winner"]})
    return pd.DataFrame(rows)


def compute_consistency(res: pd.DataFrame) -> pd.DataFrame:
    """Share of questions that got the SAME outcome in every run."""
    cons = res.groupby(["model", "approach", "qid"])["outcome"].nunique() == 1
    return cons.groupby(["model", "approach"]).mean().round(3).reset_index(name="consistency_rate")


def _row_cost(r, price_map) -> float:
    p = price_map.get(r["model"])
    if not isinstance(p, dict) or p.get("in") is None or p.get("out") is None:
        return np.nan
    return (r["tokens_in"] * p["in"] + r["tokens_out"] * p["out"]) / 1_000_000


def compute_cost_metrics(res: pd.DataFrame, custom_prices: Optional[Dict] = None) -> pd.DataFrame:
    """Cost is NaN (n/a) when no price is configured. Prices are never guessed."""
    price_map = custom_prices or PRICES
    df = res.copy()
    df["cost_usd"] = df.apply(lambda r: _row_cost(r, price_map), axis=1)
    out = df.groupby(["model", "approach"]).agg(
        total_cost=("cost_usd", lambda s: s.sum(min_count=1)),
        correct=("is_correct", "sum"),
        calls=("is_correct", "size"),
        tokens_in=("tokens_in", "sum"),
        tokens_out=("tokens_out", "sum"),
    ).reset_index()
    out["cost_per_correct"] = (out["total_cost"] / out["correct"].replace(0, np.nan)).round(5)
    out["total_cost"] = out["total_cost"].round(4)
    return out


def generate_benchmark_charts(ci_df: pd.DataFrame, out_dir: str, demo: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "overall_correct.png")
    labels = ci_df["model"] + "\n" + ci_df["approach"]
    err = [ci_df["mean"] - ci_df["ci_low"], ci_df["ci_high"] - ci_df["mean"]]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, ci_df["mean"], yerr=err, capsize=4, color="#2563eb", alpha=0.85, edgecolor="#1e40af")
    plt.ylabel("Share handled correctly (95% bootstrap CI)")
    plt.ylim(0, 1.0)
    title = "Trust Bench: overall correctness"
    if demo:
        title = "DEMO (SIMULATED DATA): " + title
    plt.title(title, fontsize=13, fontweight="bold", color="#b91c1c" if demo else "black")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def generate_pareto_chart(cost_df: pd.DataFrame, ci_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    """Cost per correct answer vs correctness. Skipped when prices are unknown."""
    m = cost_df.merge(ci_df, on=["model", "approach"])
    m = m.dropna(subset=["cost_per_correct"])
    m = m[m["cost_per_correct"] > 0]
    if m.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "cost_vs_correct.png")
    plt.figure(figsize=(8, 5))
    plt.scatter(m["cost_per_correct"], m["mean"], s=70)
    for _, r in m.iterrows():
        plt.annotate(r["model"] + "/" + r["approach"], (r["cost_per_correct"], r["mean"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Cost per correct answer (USD, log scale)")
    plt.ylabel("Share handled correctly")
    plt.title("Cost vs correctness (top-left is best)")
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def suggest_failure_type(r: pd.Series) -> str:
    """Rule-based FIRST GUESS to speed up manual labeling. Always review by hand."""
    outcome, q = r["outcome"], str(r.get("question", "")).lower()
    code = str(r.get("code", "") or "")
    if outcome == "hallucinated":
        return "hallucinated_answer"
    if outcome == "over_abstain":
        return "over_cautious"
    if outcome == "error":
        return "code_error" if str(r.get("error", "")).strip() else "format_error"
    if r["approach"] == "direct":
        return "arithmetic_or_counting_error"
    wants_filter = "excluding" in q or "does not start with c" in q
    if wants_filter and "startswith" not in code:
        return "forgot_filter"
    if "missing" in q and not any(k in code for k in ("isna", "isnull", "dropna", "notna")):
        return "ignored_missing"
    return "needs_manual_review"
