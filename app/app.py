"""Trust Bench dashboard (Streamlit). Run from the project root:
    streamlit run app/app.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustbench.config import results_paths                      # noqa: E402
from trustbench.data_engine import table_meta                     # noqa: E402
from trustbench.stats import (all_paired_differences, bootstrap_ci,             # noqa: E402
                              compute_confidence_intervals, compute_consistency,
                              compute_cost_metrics)

def grouped_chart(ci):
    """Grouped bars (one group per model, one bar per approach) with 95% intervals, scale 0 to 1."""
    models = list(ci["model"].unique())
    apps = list(ci["approach"].unique())
    width = 0.8 / max(len(apps), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, a in enumerate(apps):
        sub = ci[ci["approach"] == a].set_index("model").reindex(models)
        x = np.arange(len(models)) + j * width - 0.4 + width / 2
        ok = sub["mean"].notna().values
        if not ok.any():
            continue
        mean = sub["mean"].values[ok]
        err = [mean - sub["ci_low"].values[ok], sub["ci_high"].values[ok] - mean]
        ax.bar(x[ok], mean, width * 0.92, yerr=err, capsize=3, label=a)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share handled correctly (95% CI)")
    ax.legend(title="approach", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=max(len(apps), 1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def trap_table(view):
    """Trap questions (refusal expected) with 95% bootstrap intervals, all and by type."""
    rows = []
    t = view[view["kind"] == "abstain"]
    for (m, a), g in t.groupby(["model", "approach"]):
        for label, sub in (("all traps", g), ("unanswerable", g[g["category"] == "unanswerable"]),
                           ("ambiguous", g[g["category"] == "ambiguous"])):
            if sub.empty:
                continue
            pq = sub.groupby("qid")["is_correct"].mean()
            lo, hi = bootstrap_ci(pq.values)
            rows.append({"model": m, "approach": a, "trap_type": label, "n_questions": len(pq),
                         "refused": round(float(pq.mean()), 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    return pd.DataFrame(rows)


def metric_table(view):
    ans = view[view["kind"] != "abstain"]
    trp = view[view["kind"] == "abstain"]
    g = ["model", "approach"]
    out = pd.DataFrame({
        "overall": view.groupby(g)["is_correct"].mean(),
        "answerable_correct": ans.groupby(g)["is_correct"].mean(),
        "trap_refused": trp.groupby(g)["is_correct"].mean(),
        "hallucination_on_traps": trp.groupby(g)["outcome"].apply(lambda x: (x == "hallucinated").mean()),
        "over_abstain": ans.groupby(g)["outcome"].apply(lambda x: (x == "over_abstain").mean()),
    }).round(3)
    return out.reset_index()


st.set_page_config(page_title="Trust Bench", layout="wide")
st.title("Trust Bench: can you trust LLMs on business data?")

available = [m for m in ("real", "demo") if os.path.exists(results_paths(m)["scored"])]
if not available:
    st.warning("No results yet. Run `python run_pipeline.py --demo` (pipeline test) or a real run first.")
    st.stop()

mode = st.sidebar.radio("Results set", available, index=0,
                        help="'real' = results from real models. 'demo' = simulator, pipeline test only.")
paths = results_paths(mode)

if mode == "demo":
    st.error("DEMO / SIMULATED DATA. These numbers come from a scripted simulator, not from real models. "
             "Do not quote them anywhere.")

meta = {}
if os.path.exists(paths["meta"]):
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
current_table = table_meta()
if meta and meta.get("table_sha") != current_table.get("sha"):
    st.warning("The benchmark table changed after this run. Re-run the benchmark before trusting these results.")


@st.cache_data
def load(path, mtime):
    return pd.read_csv(path)


df = load(paths["scored"], os.path.getmtime(paths["scored"]))

with st.sidebar:
    st.header("Filters")
    models = sorted(df["model"].unique())
    approaches = sorted(df["approach"].unique())
    cats = sorted(df["category"].dropna().unique())
    sel_m = st.multiselect("Models", models, default=models)
    sel_a = st.multiselect("Approaches", approaches, default=approaches)
    sel_c = st.multiselect("Categories", cats, default=cats)
    if meta:
        st.divider()
        st.caption("Run finished: " + str(meta.get("finished")))
        st.caption("Table source: " + str(meta.get("table_source")))
        for k, v in meta.get("models", {}).items():
            st.caption(k + " = " + str(v.get("model")))

view = df[df["model"].isin(sel_m) & df["approach"].isin(sel_a) & df["category"].isin(sel_c)]
if view.empty:
    st.warning("No rows match the filters.")
    st.stop()

answerable = view[view["kind"] != "abstain"]
traps = view[view["kind"] == "abstain"]


def pct(x):
    return "n/a" if pd.isna(x) else "%.1f%%" % (x * 100)


c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall correct", pct(view["is_correct"].mean()))
c2.metric("Answerable questions correct", pct(answerable["is_correct"].mean()) if len(answerable) else "n/a")
c3.metric("Trap questions refused", pct(traps["is_correct"].mean()) if len(traps) else "n/a")
c4.metric("Hallucination on traps", pct((traps["outcome"] == "hallucinated").mean()) if len(traps) else "n/a")
st.caption("The four numbers above combine every selected model and approach. For per-model results use the table in the Overview tab.")
st.caption("%d questions, %d runs, %d scored answers shown" % (
    view["qid"].nunique(), view["run"].nunique(), len(view)))

t1, t2, t3, t4, t5 = st.tabs(["Overview", "By category", "Failures", "Cost", "Raw data"])

with t1:
    ci = compute_confidence_intervals(view)
    st.dataframe(ci, width="stretch")
    st.pyplot(grouped_chart(ci))
    st.subheader("Key metrics per model and approach")
    st.caption("Use this table, not the four headline numbers above (those mix every model and approach).")
    st.dataframe(metric_table(view), width="stretch")
    st.subheader("Trap questions with 95% confidence intervals")
    st.caption("Few trap questions means wide intervals. Read small differences as noise.")
    st.dataframe(trap_table(view), width="stretch")
    st.subheader("Run-to-run consistency")
    st.dataframe(compute_consistency(view), width="stretch")
    if len(ci) > 1:
        st.subheader("Paired comparison (same questions)")
        st.caption("If clear_winner is False, the difference may be chance. Do not claim a winner.")
        st.dataframe(all_paired_differences(view), width="stretch")

with t2:
    st.dataframe(view.pivot_table(index=["model", "approach"], columns="category",
                                  values="is_correct", aggfunc="mean").round(2), width="stretch")
    st.dataframe(view.pivot_table(index=["model", "approach"], columns="difficulty",
                                  values="is_correct", aggfunc="mean").round(2), width="stretch")

with t3:
    fails = view[~view["is_correct"]]
    outcomes = ["ALL"] + sorted(fails["outcome"].unique().tolist())
    pick = st.selectbox("Outcome", outcomes)
    if pick != "ALL":
        fails = fails[fails["outcome"] == pick]
    st.caption("%d failures (showing the first 30)" % len(fails))
    for _, row in fails.head(30).iterrows():
        with st.expander("[%s | %s] %s: %s (%s)" % (row["model"], row["approach"], row["qid"],
                                                    row["question"], row["outcome"])):
            shown = row["answer"]
            if str(shown).strip().lower() in ("nan", "none", ""):
                shown = "(refused or no answer)"
            st.write("Expected:", row["expected"], "| Model answer:", shown)
            if isinstance(row.get("code"), str) and row["code"].strip():
                st.code(row["code"], language="python")
            if isinstance(row.get("error"), str) and row["error"].strip():
                st.error(row["error"])

with t4:
    cost = compute_cost_metrics(view)
    cost_show = cost.copy()
    for col in ("total_cost", "cost_per_correct"):
        cost_show[col] = cost_show[col].apply(lambda v: "n/a" if pd.isna(v) else str(v))
    st.dataframe(cost_show, width="stretch")
    st.caption("n/a means no price is configured for that model (models.local.json). Prices are never guessed.")

with t5:
    cols = ["model", "approach", "run", "qid", "category", "difficulty", "question",
            "expected", "answer", "outcome", "latency"]
    st.dataframe(view[cols].astype(str), width="stretch")
