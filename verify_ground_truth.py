"""Independent check of the ground truth in questions/questions.json.

Every question is recomputed here with separate formulas (not the code in
trustbench/questions_engine.py) and compared with the saved expected answer.
Run from the project folder:   python verify_ground_truth.py
This is a software cross-check. It does not replace checking a few answers by hand in Excel.
"""

import json
import re
import sys

import pandas as pd

TABLE = "data/processed/benchmark_table.csv"
QUESTIONS = "questions/questions.json"

df = pd.read_csv(TABLE, parse_dates=["order_datetime"],
                 dtype={"order_id": str, "product_code": str, "customer_id": "Int64"})
df["rev"] = df["qty"] * df["unit_price"]
cancelled = df["order_id"].str.startswith("C")
ok_orders = df[~cancelled]
month = df["order_datetime"].dt.strftime("%Y-%m")
ok_month = ok_orders["order_datetime"].dt.strftime("%Y-%m")


def top_one(series):
    """Returns the label with the highest value, or 'TIE' if two labels share the top."""
    s = series.sort_values(ascending=False)
    if len(s) > 1 and s.iloc[0] == s.iloc[1]:
        return "TIE"
    return s.index[0]


# (regex, function(match) -> value). The first matching pattern is used.
RULES = [
    (r"^How many rows are in the table\?$", lambda m: len(df)),
    (r"^How many distinct order_id values are there\?$", lambda m: len(set(df["order_id"]))),
    (r"^How many distinct countries appear", lambda m: len(set(df["country"]))),
    (r"^How many distinct customer_id values are there", lambda m: df["customer_id"].dropna().nunique()),
    (r"^How many rows have a missing customer_id\?$", lambda m: int(df["customer_id"].isna().sum())),
    (r"^How many rows have a missing product_name\?$", lambda m: int(df["product_name"].isna().sum())),
    (r"^What percentage of rows have a missing customer_id",
     lambda m: round(100.0 * df["customer_id"].isna().sum() / len(df), 1)),
    (r"^How many rows are exact duplicates", lambda m: len(df) - len(df.drop_duplicates())),
    (r"^How many distinct order_id values start with the letter C",
     lambda m: len({o for o in df["order_id"] if o[0] == "C"})),
    (r"^What is the total qty of all rows where qty is negative",
     lambda m: df.loc[df["qty"] < 0, "qty"].sum()),
    (r"^What is the total revenue over all rows\?$", lambda m: df["rev"].sum()),
    (r"^What is the total revenue, excluding rows whose order_id starts with C",
     lambda m: ok_orders["rev"].sum()),
    (r"^What is the average unit_price across all rows", lambda m: df["unit_price"].sum() / len(df)),
    (r"^What is the median qty of rows where qty is greater than 0",
     lambda m: df.loc[df["qty"] > 0, "qty"].median()),
    (r"^Which country has the most rows", lambda m: top_one(df["country"].value_counts())),
    (r"^Excluding cancelled orders \(order_id starting with C\) and rows with a missing customer_id",
     lambda m: top_one(ok_orders.dropna(subset=["customer_id"]).groupby("customer_id")["rev"].sum())),
    (r"^Using only orders whose order_id does not start with C, what is the average revenue per",
     lambda m: ok_orders.groupby("order_id")["rev"].sum().mean()),
    (r"^Excluding cancelled orders, which product_code has the highest total qty",
     lambda m: top_one(ok_orders.groupby("product_code")["qty"].sum())),
    (r"^Excluding cancelled orders, in which month",
     lambda m: top_one(ok_orders.groupby(ok_month)["rev"].sum())),
    (r"^How many distinct order_id values contain 5 or more rows",
     lambda m: int((df["order_id"].value_counts() >= 5).sum())),
    (r"^Excluding cancelled orders, what is the total revenue from rows with a missing customer_id",
     lambda m: ok_orders.loc[ok_orders["customer_id"].isna(), "rev"].sum()),
    (r"^Excluding cancelled orders, what is the difference between the highest and the lowest",
     lambda m: ok_orders.groupby("country")["rev"].sum().max() - ok_orders.groupby("country")["rev"].sum().min()),
    (r"^How many rows have country equal to '(.+)'\?$", lambda m: int((df["country"] == m.group(1)).sum())),
    (r"^What is the total revenue of all rows with country equal to '(.+)'\?$",
     lambda m: df.loc[df["country"] == m.group(1), "rev"].sum()),
    (r"^What is the total qty over all rows with product_code '(.+)'\?$",
     lambda m: df.loc[df["product_code"] == m.group(1), "qty"].sum()),
    (r"^How many distinct order_id values does customer_id (\d+) have\?$",
     lambda m: df.loc[df["customer_id"] == int(m.group(1)), "order_id"].nunique()),
    (r"^What is the total revenue of all rows with order_datetime in the month (\d{4}-\d{2})",
     lambda m: df.loc[month == m.group(1), "rev"].sum()),
    (r"^What is the total revenue of all rows with product_code '(.+)'\?$",
     lambda m: df.loc[df["product_code"] == m.group(1), "rev"].sum()),
    (r"^How many rows have a unit_price greater than (\d+)",
     lambda m: int((df["unit_price"] > float(m.group(1))).sum())),
]


def same(expected, mine, kind):
    if kind == "text":
        return str(expected).strip().lower() == str(mine).strip().lower()
    e, a = float(expected), float(mine)
    return abs(e - a) <= max(0.011, 0.0005 * abs(e))


def main():
    qs = json.load(open(QUESTIONS, encoding="utf-8"))
    years = set(df["order_datetime"].dt.year.unique().tolist())
    n_ok = n_bad = n_skip = 0
    for q in qs:
        text, kind, exp = q["question"], q["kind"], q["expected"]
        if kind == "abstain":
            m = re.search(r"in the year (\d{4})", text)
            fine = exp == "CANNOT_ANSWER" and not (m and int(m.group(1)) in years)
            print(("OK      " if fine else "MISMATCH"), q["id"], "(trap) ", text[:70])
            n_ok += fine
            n_bad += (not fine)
            continue
        for pattern, fn in RULES:
            m = re.match(pattern, text)
            if m:
                mine = fn(m)
                if isinstance(mine, str) and mine == "TIE":
                    print("SKIP    ", q["id"], "tie for first place, check by hand:", text[:60])
                    n_skip += 1
                elif same(exp, mine, kind):
                    print("OK      ", q["id"], text[:60], "| expected", exp, "| mine", mine)
                    n_ok += 1
                else:
                    print("MISMATCH", q["id"], text[:60], "| expected", exp, "| mine", mine)
                    n_bad += 1
                break
        else:
            print("SKIP    ", q["id"], "no independent formula for:", text[:60])
            n_skip += 1

    print("\nChecked %d questions: %d OK, %d MISMATCH, %d skipped" % (len(qs), n_ok, n_bad, n_skip))
    if n_bad:
        print("Send me the MISMATCH lines.")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
