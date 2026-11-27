"""Model calling for Trust Bench.

Providers: Anthropic, OpenAI, Google, Ollama (local) and a scripted simulator.

Important rules in this file:
- A real model NEVER silently falls back to the simulator. Missing key, wrong
  model name or an unreachable Ollama server raises a clear error.
- The simulator is used only for models whose provider is "mock"
  (demo-sim-fast, demo-sim-reasoning). Its rows are tagged simulated=True.
- No temperature is sent, because some newer models reject it. Run-to-run
  variation is measured by running every question several times.
"""

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from trustbench.config import MODELS, prompt_check, prompt_code, prompt_direct
from trustbench.executor import clean_value, code_answer_to_dict, extract_code, run_code

load_dotenv()

_clients: Dict[str, Any] = {}

KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}


class ModelConfigError(RuntimeError):
    """Raised for missing keys, placeholder model names, unknown models."""


def _get_client(provider: str) -> Any:
    if provider in _clients:
        return _clients[provider]
    key = os.getenv(KEY_VARS[provider])
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=key)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
    elif provider == "google":
        from google import genai
        client = genai.Client(api_key=key)
    else:
        raise ModelConfigError("Unknown provider: " + provider)
    _clients[provider] = client
    return client


def parse_json_answer(text: str) -> Dict[str, Any]:
    if not text:
        return {"cannot_answer": None, "answer": None, "parse_error": True}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"cannot_answer": None, "answer": None, "parse_error": True}
    raw_json = m.group(0).strip()
    try:
        d = json.loads(raw_json)
        cannot = bool(d.get("cannot_answer", False))
        _a = d.get("answer")
        if isinstance(_a, str) and _a.strip().upper() == "CANNOT_ANSWER":
            cannot = True
        return {"cannot_answer": cannot,
                "answer": None if cannot else clean_value(d.get("answer")),
                "parse_error": False}
    except Exception:
        ca = re.search(r'"cannot_answer"\s*:\s*(true|false)', raw_json, re.IGNORECASE)
        an = re.search(r'"answer"\s*:\s*([^,\}\n]+)', raw_json)
        if ca:
            cannot = ca.group(1).lower() == "true"
            val = an.group(1).strip(' "\'') if an else None
            return {"cannot_answer": cannot,
                    "answer": None if cannot else clean_value(val),
                    "parse_error": False}
        return {"cannot_answer": None, "answer": None, "parse_error": True}


# ---------------------------------------------------------------
# Simulator (pipeline testing only; numbers are meaningless)
# ---------------------------------------------------------------
def _stable_rng(*parts: str) -> np.random.Generator:
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:8], 16))


TRAP_WORDS = ["average age", "shipping cost", "female", "satisfaction", "best customer",
              "typical order", "delivery company", "profit margin", "sales channel",
              "best seller", "most important market", "churn", "paypal"]


def simulate_model_response(model_key: str, prompt: str) -> Tuple[str, int, int, float]:
    q_match = re.search(r"Question:\s*(.+?)\n", prompt)
    q_text = (q_match.group(1) if q_match else "").lower()
    rng = _stable_rng(model_key, prompt[:300])
    lat = round(float(rng.uniform(0.4, 1.2)), 2)
    tin = len(prompt.split()) + 150
    tout = int(rng.integers(25, 90))
    strong = "reasoning" in model_key
    is_trap = any(w in q_text for w in TRAP_WORDS)

    if "Write Python code that answers the question" in prompt:
        if is_trap:
            if strong or rng.random() > 0.3:
                code = "```python\nanswer = 'CANNOT_ANSWER'\n```"
            else:
                code = "```python\nanswer = 42.5\n```"
        elif "how many rows are in the table" in q_text:
            code = "```python\nanswer = len(df)\n```"
        elif "missing customer_id" in q_text and "percentage" not in q_text:
            code = "```python\nanswer = int(df['customer_id'].isna().sum())\n```"
        elif "total revenue" in q_text and "excluding" in q_text:
            code = ("```python\nvalid = df[~df['order_id'].str.startswith('C')]\n"
                    "answer = round((valid['qty'] * valid['unit_price']).sum(), 2)\n```")
        elif "total revenue over all rows" in q_text:
            code = "```python\nanswer = round((df['qty'] * df['unit_price']).sum(), 2)\n```"
        elif "narnia" in q_text:
            code = "```python\nanswer = int((df['country'] == 'Narnia').sum())\n```"
        else:
            code = "```python\nanswer = round(float(df['qty'].sum()), 2)\n```"
        return code, tin, len(code.split()), lat

    if "Audit this logic carefully" in prompt:
        if is_trap:
            resp = '{"cannot_answer": true, "answer": null}' if strong else '{"cannot_answer": false, "answer": 1}'
        else:
            m = re.search(r"produced this result:\s*([^\n]+)", prompt)
            val = m.group(1).strip() if m else "0"
            resp = json.dumps({"cannot_answer": False, "answer": val})
        return resp, tin, tout, lat

    if is_trap:
        if strong and rng.random() > 0.2:
            resp = '{"cannot_answer": true, "answer": null}'
        else:
            resp = json.dumps({"cannot_answer": False, "answer": round(float(rng.uniform(25.0, 78.5)), 2)})
    else:
        resp = json.dumps({"cannot_answer": False, "answer": 1250.5})
    return resp, tin, tout, lat


# ---------------------------------------------------------------
# Real providers
# ---------------------------------------------------------------
def call_llm(model_key: str, prompt: str) -> Tuple[str, int, int, float]:
    """Returns (text, input_tokens, output_tokens, seconds). Raises on any problem."""
    if model_key not in MODELS:
        raise ModelConfigError("Model '" + model_key + "' is not in MODELS (config.py / models.local.json).")
    cfg = MODELS[model_key]
    provider, name = cfg["provider"], cfg["model"]

    if provider == "mock":
        return simulate_model_response(model_key, prompt)

    if name.startswith("PASTE_") or name.startswith("PUT_"):
        raise ModelConfigError("Model '" + model_key + "' still has a placeholder name. "
                               "Set the exact model name in models.local.json.")

    if provider == "ollama":
        import urllib.request
        body = json.dumps({"model": name, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise ModelConfigError("Ollama is not reachable or the model is not pulled: " + str(e))
        text = result.get("response", "")
        return (text, int(result.get("prompt_eval_count", 0)),
                int(result.get("eval_count", 0)), time.time() - start)

    if provider == "openai_compatible":
        base_url, key_env = cfg.get("base_url"), cfg.get("api_key_env")
        if not base_url or not key_env:
            raise ModelConfigError("Model '" + model_key + "' needs base_url and api_key_env in models.local.json.")
        api_key = os.getenv(key_env)
        if not api_key:
            raise ModelConfigError("Missing " + key_env + " in .env for model '" + model_key + "'.")
        cache_key = "oc|" + base_url + "|" + key_env
        if cache_key not in _clients:
            from openai import OpenAI
            _clients[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
        start = time.time()
        r = _clients[cache_key].chat.completions.create(
            model=name, messages=[{"role": "user", "content": prompt}])
        text = r.choices[0].message.content or ""
        usage = getattr(r, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        return text, int(tin), int(tout), time.time() - start

    if provider not in KEY_VARS:
        raise ModelConfigError("Unknown provider: " + provider)
    if not os.getenv(KEY_VARS[provider]):
        raise ModelConfigError("Missing " + KEY_VARS[provider] + " in .env for model '" + model_key + "'.")

    client = _get_client(provider)
    start = time.time()

    if provider == "openai":
        r = client.chat.completions.create(model=name, messages=[{"role": "user", "content": prompt}])
        text = r.choices[0].message.content or ""
        tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
    elif provider == "anthropic":
        r = client.messages.create(model=name, max_tokens=2000,
                                   messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
    else:
        r = client.models.generate_content(model=name, contents=prompt)
        text = r.text or ""
        um = getattr(r, "usage_metadata", None)
        tin = getattr(um, "prompt_token_count", 0) or 0
        tout = getattr(um, "candidates_token_count", 0) or 0

    return text, int(tin), int(tout), time.time() - start


def preflight(model_keys: List[str]) -> bool:
    """Sends one tiny request to each model so you find key/name problems before spending money."""
    ok_all = True
    for m in model_keys:
        try:
            text, ti, to, lat = call_llm(m, "Reply with exactly the word OK.")
            print("[preflight] OK   ", m, "->", text.strip()[:30], "| tokens", ti, "/", to)
        except Exception as e:
            ok_all = False
            print("[preflight] FAIL ", m, "->", str(e)[:200])
    return ok_all


# ---------------------------------------------------------------
# One question -> three approach rows
# ---------------------------------------------------------------
def solve_all(model_key: str, question: str, table_csv: str, head_csv: str,
              df: pd.DataFrame) -> List[Dict[str, Any]]:
    simulated = MODELS[model_key]["provider"] == "mock"
    model_name = MODELS[model_key]["model"]
    rows = []

    # 1. direct (can be switched off per model with "skip_direct": true)
    if not MODELS[model_key].get("skip_direct"):
        text, ti, to, lat = call_llm(model_key, prompt_direct(question, table_csv))
        rows.append({"approach": "direct", **parse_json_answer(text), "code": "", "error": "",
                     "raw": text, "tokens_in": ti, "tokens_out": to, "latency": lat})

    # 2. code
    ctext, ci, co, clat = call_llm(model_key, prompt_code(question, head_csv))
    code = extract_code(ctext)
    ans, err = run_code(code, df)
    rows.append({"approach": "code", **code_answer_to_dict(ans, err), "code": code,
                 "error": err or "", "raw": ctext, "tokens_in": ci, "tokens_out": co,
                 "latency": clat})

    # 3. check (reuses the code and its result from approach 2)
    result_text = ("ERROR: " + err) if err else str(clean_value(ans))
    ktext, ki, ko, klat = call_llm(model_key, prompt_check(question, head_csv, code, result_text))
    rows.append({"approach": "check", **parse_json_answer(ktext), "code": code,
                 "error": err or "", "raw": ktext, "tokens_in": ci + ki, "tokens_out": co + ko,
                 "latency": clat + klat})

    for r in rows:
        r["simulated"] = simulated
        r["model_name"] = model_name
    return rows
