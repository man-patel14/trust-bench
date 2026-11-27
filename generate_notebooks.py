"""Regenerates the 4 Jupyter notebooks. Run: python generate_notebooks.py"""

import json


def nb(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 2,
            "metadata": {"language_info": {"name": "python"},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}


def code(src):
    lines = [l + "\n" for l in src.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def md(src):
    lines = [l + "\n" for l in src.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


NB1 = [
    md("# 01 - Prepare the benchmark data\n"
       "Put `online_retail_II.xlsx` (or `.csv`) into `data/raw/` first. "
       "Without it only a clearly labelled SYNTHETIC demo table can be made."),
    code('import os\nfor f in ["data/raw", "data/processed", "questions", "results/real", "results/demo"]:\n'
         '    os.makedirs(f, exist_ok=True)\nprint(os.listdir("data/raw"))'),
    code('from dotenv import load_dotenv\nimport os\nload_dotenv()\n'
         'for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:\n    print(k, "found:", bool(os.getenv(k)))'),
    code('from trustbench.data_engine import prepare_benchmark_table, table_meta\n\n'
         '# force_recreate=True rebuilds the table. allow_synthetic=True only for pipeline demos.\n'
         'bench = prepare_benchmark_table(force_recreate=True, allow_synthetic=False)\n'
         'print("Shape:", bench.shape)\nprint(table_meta())\nbench.isna().sum()'),
    md("**Check:** `source` must say `real` before you run real models. "
       "If you changed the table, regenerate the questions in notebook 02."),
]

NB2 = [
    md("# 02 - Questions and ground truth\nGround truth is computed with pandas, never typed by hand."),
    code('from trustbench.questions_engine import generate_and_save_questions\nfrom utils import load_table\n\n'
         'df = load_table()\n# scale=2 makes a bigger question suite\nqdf = generate_and_save_questions(df, scale=1)\n'
         'print("Total questions:", len(qdf))\nqdf.groupby(["category", "difficulty"]).size()'),
    code('# Verify 10 answers by hand in Excel (open data/processed/benchmark_table.csv)\n'
         'qdf.sample(10, random_state=0)[["id", "question", "kind", "expected"]]'),
    md("**Do this:** solve the 10 questions above in Excel yourself. All 10 must match. "
       "Write down that you did it, you will need it in interviews."),
]

NB3 = [
    md("# 03 - Run the models\n"
       "Models live in `trustbench/config.py` and `models.local.json` (copy the `.example` file). "
       "Real model names and prices must be filled in by you from the provider websites."),
    code('from trustbench.models import preflight\n\n# 1 tiny call per model: finds wrong keys or model names BEFORE spending money\n'
         'preflight(["gemini-small"])'),
    code('from run_pipeline import run_benchmark\n\n'
         '# Pilot: 1 model, 1 run, 10 questions spread over all categories (includes trap questions)\n'
         'run_benchmark(models=["gemini-small"], runs=[1], question_limit=10, pause_seconds=1, force_restart=False)'),
    code('# Full run: all questions, 3 runs. Do one model at a time. Progress is saved, you can re-run safely.\n'
         '# run_benchmark(models=["gemini-small"], runs=[1, 2, 3], question_limit=None, pause_seconds=1)'),
    md("Simulator (no keys, pipeline test only, NOT real results): `run_benchmark(models=['demo-sim-fast','demo-sim-reasoning'], runs=[1,2], question_limit=30)`"),
]

NB4 = [
    md("# 04 - Scoring, statistics, failure analysis"),
    code('import pandas as pd\nfrom trustbench.config import results_paths\nfrom trustbench.scoring import score_results, compute_metrics_summary\n\n'
         'MODE = "real"   # use "demo" only to look at simulator output\n'
         'scored = score_results(MODE)\nif MODE == "demo":\n    print("SIMULATED DATA - do not quote these numbers")\ncompute_metrics_summary(scored)'),
    code('from trustbench.stats import (compute_confidence_intervals, compute_consistency, compute_cost_metrics,\n'
         '                             all_paired_differences, generate_benchmark_charts)\n\n'
         'ci = compute_confidence_intervals(scored)\nprint(ci)\n'
         'generate_benchmark_charts(ci, results_paths(MODE)["charts"], demo=(MODE == "demo"))\n'
         'print(compute_consistency(scored))\nprint(compute_cost_metrics(scored))'),
    code('# If clear_winner is False, say "no clear winner". Do not claim differences that may be chance.\nall_paired_differences(scored)'),
    md("## Failure analysis (manual)\n"
       "1. Open `results/real/failures_to_label.csv` in Excel. `suggested_type` is only a first guess.\n"
       "2. Fill the `failure_type` column by hand. Save as `failures_labeled.csv` in the same folder.\n"
       "3. Run the next cell."),
    code('lab = pd.read_csv(results_paths(MODE)["dir"] + "/failures_labeled.csv")\nlab["failure_type"] = lab["failure_type"].fillna("unlabeled")\n'
         'print(pd.crosstab([lab["model"], lab["approach"]], lab["failure_type"]))'),
]

for name, cells in [("01_prepare_data", NB1), ("02_questions", NB2), ("03_run_models", NB3), ("04_scoring", NB4)]:
    with open(name + ".ipynb", "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1)
    print("wrote", name + ".ipynb")
