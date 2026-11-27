import os
import sys

import pandas as pd

PATH = os.path.join("results", "real", "failures_labeled.csv")
OUT = os.path.join("results", "real", "failure_summary.csv")
ALLOWED = {"wrong_column", "forgot_filter", "wrong_aggregation", "ignored_missing",
           "arithmetic_or_counting_error", "hallucinated_answer", "over_cautious",
           "code_error", "format_error", "other"}

if not os.path.exists(PATH):
    sys.exit("File not found: " + PATH)

lab = None
for enc in ("utf-8-sig", "cp1252"):
    try:
        lab = pd.read_csv(PATH, encoding=enc)
        break
    except UnicodeDecodeError:
        continue
if lab is None or "failure_type" not in lab.columns:
    sys.exit("Could not read the file or the failure_type column is missing.")

lab["failure_type"] = lab["failure_type"].astype(str).str.strip()
empty = lab["failure_type"].isin(["", "nan"])
print("Rows in file:", len(lab), "| still unlabeled:", int(empty.sum()))
unknown = sorted(set(lab.loc[~empty, "failure_type"]) - ALLOWED)
if unknown:
    print("WARNING: these labels are not in the standard list:", unknown)
lab.loc[empty, "failure_type"] = "UNLABELED"

table = pd.crosstab([lab["model"], lab["approach"]], lab["failure_type"], margins=True)
print("\nFailures by model, approach and type:\n")
print(table.to_string())
table.to_csv(OUT, encoding="utf-8")

print("\nExamples (up to 2 per type and approach):")
for (ftype, appr), grp in lab.groupby(["failure_type", "approach"]):
    print("\n=== %s | %s (%d rows) ===" % (ftype, appr, len(grp)))
    for _, r in grp.head(2).iterrows():
        print(" ", r["qid"], "|", str(r["question"])[:90])
        print("     expected:", r["expected"], "| model answer:", r["answer"])
print("\nSaved:", OUT)