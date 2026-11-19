"""Spending guard for eval runs: the key's real usage from OpenRouter (GET /api/v1/key).

A budget is a ceiling on new spending from a recorded starting point, kept in
logs/eval_budget.json (git-ignored): {"start_usage": 0.31, "limit_usd": 3.0}.

    python -m evals.budget            # print usage, spent since the start, what is left
    python -m evals.budget --start 3  # record the current usage as the start, with a $3 limit
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BUDGET_PATH = ROOT / "logs" / "eval_budget.json"
KEY_URL = "https://openrouter.ai/api/v1/key"


def key_usage() -> float:
    """Total USD the API key has used so far, as OpenRouter reports it."""
    response = httpx.get(KEY_URL, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=30)
    response.raise_for_status()
    return float(response.json()["data"]["usage"])


def start(limit_usd: float) -> dict:
    data = {"start_usage": key_usage(), "limit_usd": limit_usd}
    BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_PATH.write_text(json.dumps(data), encoding="utf-8")
    return data


def status() -> dict:
    """{"usage", "spent", "left", "limit_usd"}; without a recorded start, everything is spent=0."""
    usage = key_usage()
    data = json.loads(BUDGET_PATH.read_text(encoding="utf-8")) if BUDGET_PATH.is_file() else {"start_usage": usage, "limit_usd": 0.0}
    spent = usage - data["start_usage"]
    return {"usage": usage, "spent": round(spent, 4), "left": round(data["limit_usd"] - spent, 4), "limit_usd": data["limit_usd"]}


def can_spend(estimate_usd: float) -> bool:
    return status()["left"] >= estimate_usd


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, metavar="LIMIT_USD")
    args = parser.parse_args()
    print(json.dumps(start(args.start) if args.start is not None else status()))
    sys.exit(0)
