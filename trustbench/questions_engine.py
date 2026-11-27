"""
Questions Engine for Trust Bench.
Generates 72+ ground-truth verified business questions across 6 analytical categories,
including trick questions (0 answer) and unanswerable/ambiguous trap questions.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from trustbench.config import (
    QUESTIONS_PATH,
    QUESTIONS_REVIEW_PATH,
    TABLE_PATH,
)


def unique_top(series: pd.Series) -> Optional[Any]:
    """Returns top label, or None if two or more labels tie for first."""
    s = series.sort_values(ascending=False)
    if len(s) > 1 and s.iloc[0] == s.iloc[1]:
        return None
    return s.index[0]


def build_questions_suite(df: pd.DataFrame, scale: int = 1) -> List[Dict[str, Any]]:
    """
    Constructs the exact 72+ question benchmark suite as documented in Trust Bench Step 5.
    Computes exact ground truth using pandas.
    """
    df = df.copy()
    df["revenue"] = df["qty"] * df["unit_price"]
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])
    valid = df[~df["order_id"].str.startswith("C")]  # not cancelled orders

    questions = []

    def add(category: str, difficulty: str, question: str, expected: Any, kind: str = "number"):
        """kind: 'number', 'text', or 'abstain'."""
        if expected is None:
            return  # skip questions with ties

        if kind == "number":
            expected = round(float(expected), 2)
        elif kind == "text":
            expected = str(expected).strip()

        questions.append({
            "id": "Q" + str(len(questions) + 1).zfill(3),
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "kind": kind,
            "expected": expected,
        })

    # ---------- 1. Simple Lookups and Counts (Easy) ----------
    add("lookup", "easy", "How many rows are in the table?", len(df))
    add("lookup", "easy", "How many distinct order_id values are there?", df["order_id"].nunique())
    add("lookup", "easy", "How many distinct countries appear in the table?", df["country"].nunique())
    add(
        "lookup",
        "easy",
        "How many distinct customer_id values are there? Ignore missing values.",
        df["customer_id"].nunique(),
    )
    add(
        "missing_data",
        "easy",
        "How many rows have a missing customer_id?",
        int(df["customer_id"].isna().sum()),
    )
    add(
        "missing_data",
        "easy",
        "How many rows have a missing product_name?",
        int(df["product_name"].isna().sum()),
    )
    add(
        "missing_data",
        "medium",
        "What percentage of rows have a missing customer_id? Round to 1 decimal place.",
        round(df["customer_id"].isna().mean() * 100, 1),
    )
    add(
        "messy_data",
        "medium",
        "How many rows are exact duplicates of an earlier row? Count only the extra copies.",
        int(df.duplicated().sum()),
    )
    add(
        "messy_data",
        "medium",
        "How many distinct order_id values start with the letter C?",
        df[df["order_id"].str.startswith("C")]["order_id"].nunique(),
    )
    add(
        "messy_data",
        "medium",
        "What is the total qty of all rows where qty is negative?",
        int(df[df["qty"] < 0]["qty"].sum()),
    )

    # ---------- 2. Aggregations (Medium) ----------
    add("aggregation", "medium", "What is the total revenue over all rows?", df["revenue"].sum())
    add(
        "aggregation",
        "medium",
        "What is the total revenue, excluding rows whose order_id starts with C?",
        valid["revenue"].sum(),
    )
    add(
        "aggregation",
        "easy",
        "What is the average unit_price across all rows?",
        df["unit_price"].mean(),
    )
    add(
        "aggregation",
        "medium",
        "What is the median qty of rows where qty is greater than 0?",
        df[df["qty"] > 0]["qty"].median(),
    )
    add(
        "aggregation",
        "medium",
        "Which country has the most rows? Answer with the country name.",
        df["country"].value_counts().idxmax(),
        kind="text",
    )

    # ---------- 3. Multi-Step Reasoning (Hard) ----------
    top_cust_rev = unique_top(valid.dropna(subset=["customer_id"]).groupby("customer_id")["revenue"].sum())
    if top_cust_rev is not None:
        add(
            "multi_step",
            "hard",
            "Excluding cancelled orders (order_id starting with C) and rows with a missing customer_id, "
            "which customer_id has the highest total revenue? Answer with the customer_id number.",
            int(top_cust_rev),
        )

    avg_rev_per_order = valid.groupby("order_id")["revenue"].sum().mean()
    add(
        "multi_step",
        "hard",
        "Using only orders whose order_id does not start with C, what is the average revenue per distinct order_id?",
        avg_rev_per_order,
    )

    top_prod_qty = unique_top(valid.groupby("product_code")["qty"].sum())
    if top_prod_qty is not None:
        add(
            "multi_step",
            "hard",
            "Excluding cancelled orders, which product_code has the highest total qty? Answer with the product_code.",
            top_prod_qty,
            kind="text",
        )

    top_month_rev = unique_top(valid.groupby(valid["order_datetime"].dt.strftime("%Y-%m"))["revenue"].sum())
    if top_month_rev is not None:
        add(
            "multi_step",
            "hard",
            "Excluding cancelled orders, in which month (format YYYY-MM) is total revenue the highest?",
            top_month_rev,
            kind="text",
        )

    add(
        "multi_step",
        "hard",
        "How many distinct order_id values contain 5 or more rows?",
        int((df.groupby("order_id").size() >= 5).sum()),
    )
    add(
        "multi_step",
        "hard",
        "Excluding cancelled orders, what is the total revenue from rows with a missing customer_id?",
        valid[valid["customer_id"].isna()]["revenue"].sum(),
    )

    c_rev = valid.groupby("country")["revenue"].sum()
    if len(c_rev) > 0:
        add(
            "multi_step",
            "hard",
            "Excluding cancelled orders, what is the difference between the highest and the lowest total revenue of any country?",
            c_rev.max() - c_rev.min(),
        )

    # ---------- 4. Templated Questions ----------
    top_countries = df["country"].value_counts().head(6 * scale).index.tolist()
    for c in top_countries:
        add("lookup", "easy", f"How many rows have country equal to '{c}'?", (df["country"] == c).sum())
        add(
            "aggregation",
            "medium",
            f"What is the total revenue of all rows with country equal to '{c}'?",
            df[df["country"] == c]["revenue"].sum(),
        )

    codes = df["product_code"].drop_duplicates().sample(min(8 * scale, df["product_code"].nunique()), random_state=1).tolist()
    for p in codes:
        add(
            "aggregation",
            "medium",
            f"What is the total qty over all rows with product_code '{p}'?",
            df[df["product_code"] == p]["qty"].sum(),
        )

    custs = df["customer_id"].dropna().drop_duplicates().sample(min(8 * scale, df["customer_id"].nunique()), random_state=1).tolist()
    for cid in custs:
        add(
            "aggregation",
            "medium",
            f"How many distinct order_id values does customer_id {int(cid)} have?",
            df[df["customer_id"] == cid]["order_id"].nunique(),
        )

    months = sorted(df["order_datetime"].dt.strftime("%Y-%m").unique().tolist())[:8 * scale]
    for m in months:
        add(
            "aggregation",
            "medium",
            f"What is the total revenue of all rows with order_datetime in the month {m} (YYYY-MM)?",
            df[df["order_datetime"].dt.strftime("%Y-%m") == m]["revenue"].sum(),
        )

    # ---------- 5. Trick Questions with Correct Answer = 0 (False Premise) ----------
    add("false_premise", "medium", "How many rows have country equal to 'Narnia'?", 0)
    add("false_premise", "medium", "What is the total revenue of all rows with product_code 'ZZZ999'?", 0)
    add("false_premise", "medium", "How many rows have a unit_price greater than 10000?", 0)

    # ---------- 6. Questions that must NOT be answered (Traps: Unanswerable & Ambiguous) ----------
    far_year = int(df["order_datetime"].dt.year.max()) + 3
    traps = [
        ("unanswerable", "What is the average age of the customers?"),
        ("unanswerable", "What is the profit margin of the product with the highest revenue?"),
        ("unanswerable", "How many of the customers are female?"),
        ("unanswerable", "Which sales channel (website, app or store) produced the most revenue?"),
        ("unanswerable", "What is the average customer satisfaction score for orders from France?"),
        ("unanswerable", "What is the total shipping cost of all orders?"),
        ("unanswerable", f"What was the average revenue per order in the year {far_year}?"),
        ("unanswerable", "Which delivery company delivered the most orders?"),
        ("ambiguous", "Who is our best customer?"),
        ("ambiguous", "Which product is the best seller?"),
        ("ambiguous", "Which country is the most important market?"),
        ("ambiguous", "What is the typical order size?"),
        ("ambiguous", "What was the overall customer churn rate?"),
        ("unanswerable", "What percentage of orders were paid using PayPal?"),
    ]

    for cat, q in traps:
        add(cat, "hard", q, "CANNOT_ANSWER", kind="abstain")

    # ---- 16 more trap questions (v1.1). Written before any model saw them. ----
    vc = df["product_code"].value_counts()
    top_code = sorted(vc.index, key=lambda c: (-int(vc[c]), c))[0]
    low_cust = int(df["customer_id"].dropna().min())
    extra_traps = [
        # unanswerable: the table has no such information
        ("unanswerable", "What is the average discount percentage given on orders?"),
        ("unanswerable", "Which warehouse shipped the most orders?"),
        ("unanswerable", "What is the email address of customer_id " + str(low_cust) + "?"),
        ("unanswerable", "What is the current stock level of product_code '" + str(top_code) + "'?"),
        ("unanswerable", "Which sales representative handled the largest order?"),
        ("unanswerable", "What was the total marketing spend over the whole period?"),
        ("unanswerable", "Which product category generated the most revenue?"),
        ("unanswerable", "What is the average delivery time in days from order to arrival?"),
        # ambiguous: no single correct answer without a definition
        ("ambiguous", "Which month was the best for the business?"),
        ("ambiguous", "Which country is growing the fastest?"),
        ("ambiguous", "Who are our loyal customers?"),
        ("ambiguous", "How many high-value customers do we have?"),
        ("ambiguous", "Which orders are unusually large?"),
        ("ambiguous", "How healthy is the business?"),
        ("ambiguous", "Which customers are most at risk of leaving?"),
        ("ambiguous", "Which products are doing well?"),
    ]
    for cat, q in extra_traps:
        add(cat, "hard", q, "CANNOT_ANSWER", kind="abstain")

    return questions


def generate_and_save_questions(
    df: Optional[pd.DataFrame] = None,
    table_path: str = TABLE_PATH,
    out_json: str = QUESTIONS_PATH,
    out_csv: str = QUESTIONS_REVIEW_PATH,
    scale: int = 1,
) -> pd.DataFrame:
    """Generates the questions suite and saves to JSON and CSV."""
    if df is None:
        df = pd.read_csv(
            table_path,
            parse_dates=["order_datetime"],
            dtype={"order_id": str, "product_code": str, "customer_id": "Int64"},
        )

    questions = build_questions_suite(df, scale=scale)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    qdf = pd.DataFrame(questions)
    qdf.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[TrustBench] Generated {len(qdf)} questions saved to {out_json} and {out_csv}")
    print(qdf.groupby(["category", "difficulty"]).size())
    return qdf


if __name__ == "__main__":
    generate_and_save_questions()



def select_questions(questions: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    """Picks `limit` questions spread evenly over the whole list.
    Trap questions sit at the end of the list, so a plain questions[:limit]
    would silently drop all of them."""
    if limit is None or limit >= len(questions):
        return list(questions)
    idx = sorted(set(np.linspace(0, len(questions) - 1, limit).round().astype(int).tolist()))
    return [questions[i] for i in idx]
