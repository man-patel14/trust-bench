"""Trust Bench command line runner.

Examples
  python run_pipeline.py --download-data                # download real UCI data + build table
  python run_pipeline.py --prepare-data                 # build table from data/raw (real data)
  python run_pipeline.py --prepare-data --allow-synthetic   # demo table only
  python run_pipeline.py --preflight --models claude-haiku  # 1 tiny call per model
  python run_pipeline.py --demo                         # simulator, no keys (pipeline test only)
  python run_pipeline.py --models claude-haiku,claude-sonnet --runs 1,2,3
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import pandas as pd

from trustbench.config import MODELS, QUESTIONS_PATH, TABLE_PATH, is_demo_model, results_paths
from trustbench.data_engine import (download_uci_dataset, file_sha256, prepare_benchmark_table,
                                    table_meta)
from trustbench.models import ModelConfigError, preflight, solve_all
from trustbench.questions_engine import generate_and_save_questions, select_questions
from trustbench.report import generate_pdf_report
from trustbench.scoring import compute_metrics_summary, score_results
from trustbench.stats import (all_paired_differences, compute_confidence_intervals,
                              compute_consistency, compute_cost_metrics,
                              generate_benchmark_charts, generate_pareto_chart,
                              suggest_failure_type)


def run_benchmark(models, runs, question_limit, pause_seconds=0.5, force_restart=False):
    demo_flags = [is_demo_model(m) for m in models]
    if any(demo_flags) and not all(demo_flags):
        sys.exit("Do not mix simulator and real models in one run. Run them separately.")
    mode = "demo" if all(demo_flags) else "real"
    paths = results_paths(mode)
    os.makedirs(paths["dir"], exist_ok=True)

    meta = table_meta()
    if mode == "real" and meta.get("source") != "real":
        sys.exit("The benchmark table is '%s', not real UCI data. Real model runs are blocked.\n"
                 "Put online_retail_II.xlsx/.csv in data/raw and run: python run_pipeline.py --prepare-data --force"
                 % meta.get("source"))

    print("=" * 70)
    print("TRUST BENCH | mode:", mode.upper(), "| table source:", meta.get("source"))
    print("=" * 70)

    df = prepare_benchmark_table(allow_synthetic=(mode == "demo"))
    if not os.path.exists(QUESTIONS_PATH):
        generate_and_save_questions(df)
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    eval_questions = select_questions(questions, question_limit)
    n_traps = sum(q["kind"] == "abstain" for q in eval_questions)
    print("Questions: %d (%d trap) | runs: %s | models: %s" % (len(eval_questions), n_traps, runs, models))

    if force_restart and os.path.exists(paths["raw"]):
        os.remove(paths["raw"])

    table_csv = open(TABLE_PATH, encoding="utf-8").read()
    head_csv = df.head(5).to_csv(index=False)

    done = set()
    if os.path.exists(paths["raw"]):
        for line in open(paths["raw"], encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["qid"], r["run"]))

    failures = 0
    for model_key in models:
        print("\n--> Model:", model_key, "(" + MODELS[model_key]["model"] + ")")
        for run_id in runs:
            for q in eval_questions:
                if (model_key, q["id"], run_id) in done:
                    continue
                try:
                    rows = solve_all(model_key, q["question"], table_csv, head_csv, df)
                except ModelConfigError as e:
                    sys.exit("\nConfiguration problem, stopping: " + str(e))
                except Exception as e:
                    failures += 1
                    print("\n[API error] %s %s: %s" % (model_key, q["id"], str(e)[:120]))
                    if failures >= 10:
                        sys.exit("Too many API errors, stopping. Fix the problem and run again; progress is saved.")
                    time.sleep(3)
                    continue
                with open(paths["raw"], "a", encoding="utf-8") as f:
                    for r in rows:
                        r.update({"model": model_key, "run": run_id, "qid": q["id"]})
                        f.write(json.dumps(r, default=str) + "\n")
                sys.stdout.write("\r%s run %s %s saved" % (model_key, run_id, q["id"]))
                sys.stdout.flush()
                time.sleep(pause_seconds)

    print("\n\nScoring...")
    scored = score_results(mode)
    print(compute_metrics_summary(scored).to_string())

    ci = compute_confidence_intervals(scored)
    cost = compute_cost_metrics(scored)
    chart = generate_benchmark_charts(ci, paths["charts"], demo=(mode == "demo"))
    generate_pareto_chart(cost, ci, paths["charts"])
    print("\nConsistency:\n", compute_consistency(scored).to_string(index=False))
    print("\nCost:\n", cost.to_string(index=False))
    if len(ci) > 1:
        print("\nPaired comparisons (clear_winner=False means no clear winner):")
        print(all_paired_differences(scored).to_string(index=False))

    fails = scored[~scored["is_correct"]].copy()
    fails["suggested_type"] = fails.apply(suggest_failure_type, axis=1)
    fails["failure_type"] = ""
    fails.to_csv(paths["failures"], index=False, encoding="utf-8")

    run_meta = {
        "mode": mode, "finished": datetime.now().isoformat(timespec="seconds"),
        "table_source": meta.get("source"), "table_sha": file_sha256(TABLE_PATH),
        "models": {m: MODELS[m] for m in models}, "runs": runs,
        "n_questions": len(eval_questions), "n_trap_questions": n_traps,
        "python": platform.python_version(),
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    pdf = generate_pdf_report(mode)
    print("\nDone. Chart: %s\nReport: %s\nFailures to label: %s" % (chart, pdf, paths["failures"]))
    print("Dashboard: streamlit run app/app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trust Bench")
    ap.add_argument("--download-data", action="store_true", help="download UCI Online Retail II into data/raw (needs internet)")
    ap.add_argument("--prepare-data", action="store_true", help="build the benchmark table from data/raw")
    ap.add_argument("--allow-synthetic", action="store_true", help="with --prepare-data: make a DEMO synthetic table")
    ap.add_argument("--force", action="store_true", help="rebuild table / restart results from scratch")
    ap.add_argument("--build-questions", action="store_true", help="(re)generate questions.json")
    ap.add_argument("--scale", type=int, default=1, help="question suite size multiplier (1 or 2)")
    ap.add_argument("--preflight", action="store_true", help="1 tiny call per model to check keys and names")
    ap.add_argument("--demo", action="store_true", help="simulator run (pipeline test, NOT real results)")
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--runs", type=str, default="1,2,3")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only N questions (spread evenly)")
    ap.add_argument("--pause", type=float, default=0.5)
    a = ap.parse_args()

    if a.download_data:
        if download_uci_dataset():
            prepare_benchmark_table(force_recreate=True)
            generate_and_save_questions(scale=a.scale)
            print("Real data ready. Delete results/real if it has older results, then run your models.")
        sys.exit(0)
    if a.prepare_data:
        prepare_benchmark_table(force_recreate=a.force, allow_synthetic=a.allow_synthetic)
        generate_and_save_questions(scale=a.scale)
        sys.exit(0)
    if a.build_questions:
        generate_and_save_questions(scale=a.scale)
        sys.exit(0)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    runs = [int(x) for x in a.runs.split(",") if x.strip()]
    if a.demo:
        models = ["demo-sim-fast", "demo-sim-reasoning"]
        runs, a.pause = [1, 2], 0.0
        a.limit = a.limit or 30
    if not models:
        sys.exit("Choose models with --models name1,name2 (see MODELS in trustbench/config.py) or use --demo.")
    for m in models:
        if m not in MODELS:
            sys.exit("Unknown model key: " + m)
    if a.preflight:
        sys.exit(0 if preflight(models) else 1)
    run_benchmark(models, runs, a.limit, a.pause, a.force)
