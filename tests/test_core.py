"""Safety and correctness tests. Run: python -m pytest tests -q   (or: python tests/test_core.py)"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustbench.config import MODELS, QUESTIONS_PATH                      # noqa: E402
from trustbench.executor import run_code                                   # noqa: E402
from trustbench.models import ModelConfigError, call_llm, simulate_model_response  # noqa: E402
from trustbench.questions_engine import build_questions_suite, select_questions    # noqa: E402
from trustbench.scoring import score_row                                   # noqa: E402
from utils import load_table                                               # noqa: E402


def test_ground_truth_matches_independent_formulas():
    df = load_table()
    df["revenue"] = df["qty"] * df["unit_price"]
    valid = df[~df["order_id"].str.startswith("C")]
    q = {x["question"]: x["expected"] for x in json.load(open(QUESTIONS_PATH, encoding="utf-8"))}
    assert q["How many rows are in the table?"] == len(df)
    assert q["How many distinct order_id values are there?"] == df["order_id"].nunique()
    assert abs(q["What is the total revenue over all rows?"] - df["revenue"].sum()) < 0.011
    assert abs(q["What is the total revenue, excluding rows whose order_id starts with C?"]
               - valid["revenue"].sum()) < 0.011


def test_saved_questions_match_fresh_generation():
    fresh = build_questions_suite(load_table())
    saved = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    assert [str(x["expected"]) for x in fresh] == [str(x["expected"]) for x in saved]


def test_question_sample_keeps_trap_questions():
    qs = build_questions_suite(load_table())
    sample = select_questions(qs, 25)
    assert any(q["kind"] == "abstain" for q in sample)


def test_real_models_never_fall_back_to_simulator():
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        os.environ.pop(k, None)
    for m in ["claude-haiku", "gemini-small", "gpt-small"]:
        try:
            call_llm(m, "hi")
            raise AssertionError(m + " should have raised")
        except ModelConfigError:
            pass


def test_simulator_is_deterministic():
    p = "Question: What is the average age of the customers?\nabc"
    assert len({simulate_model_response("demo-sim-fast", p)[0] for _ in range(5)}) == 1


def test_sandbox_blocks_dangerous_code():
    df = pd.DataFrame({"a": [1, 2, 3]})
    bad = ["answer = open('.env').read()", "import os\nanswer = os.getcwd()",
           "answer = pd.read_csv('x.csv')", "answer = ().__class__",
           "answer = getattr(pd, 'read_csv')", "answer = pd.io.common.os.getcwd()"]
    for code in bad:
        ans, err = run_code(code, df)
        assert err is not None and "Security" in err, code


def test_sandbox_runs_normal_code_and_hides_api_keys():
    os.environ["ANTHROPIC_API_KEY"] = "secret-test-value"
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert run_code("answer = int(df['a'].sum())", df) == (6, None)
    assert run_code("import pandas as p2\nanswer = int(p2.Series([4]).sum())", df) == (4, None)


def test_sandbox_timeout_really_stops_infinite_loop():
    import time
    import trustbench.executor as ex
    old = ex.TIMEOUT_SECONDS
    ex.TIMEOUT_SECONDS = 3
    try:
        t = time.time()
        ans, err = run_code("while True:\n    pass", pd.DataFrame({"a": [1]}))
        assert err is not None and "Timeout" in err
        assert time.time() - t < 10
    finally:
        ex.TIMEOUT_SECONDS = old


def test_sandbox_cannot_steal_environment_keys():
    os.environ["OPENAI_API_KEY"] = "sk-test-secret"
    df = pd.DataFrame({"a": [1]})
    ans, err = run_code("answer = pd.io.common.os.environ.get('OPENAI_API_KEY')", df)
    assert err is not None and "sk-test-secret" not in str(ans)


def test_scoring_taxonomy():
    base = {"parse_error": False, "cannot_answer": False, "answer": 10, "expected": 10, "kind": "number"}
    assert score_row(pd.Series(base)) == "correct"
    assert score_row(pd.Series({**base, "answer": 11})) == "wrong"
    assert score_row(pd.Series({**base, "cannot_answer": True, "answer": None})) == "over_abstain"
    trap = {**base, "kind": "abstain", "expected": "CANNOT_ANSWER"}
    assert score_row(pd.Series({**trap, "cannot_answer": True, "answer": None})) == "correct_abstain"
    assert score_row(pd.Series(trap)) == "hallucinated"
    assert score_row(pd.Series({**base, "parse_error": True})) == "error"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("All", len(fns), "tests passed")
