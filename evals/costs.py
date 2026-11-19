"""Cost helpers for the evals: OpenRouter prices and the theoretical cost of free models.

Actual cost comes from OpenRouter's usage.cost on every reply. The theoretical cost of a free
model is its tokens times the price of the paid model that does the same job: the balanced
profile's first model of the same role in agent/models.yaml (writer -> Sonnet, checker ->
Gemini Flash, small -> Gemini Flash Lite). The role of a call is taken from its node.
Prices are fetched from /api/v1/models and saved to evals/results/prices.json.
"""

import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "evals" / "results" / "prices.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"

NODE_ROLES = {
    "evaluator": "checker",
    "judge": "checker",
    "scope_check": "small",
    "classify_type": "small",
    "key_points": "small",
    "injection_guard": "small",
    "gate_topics": "small",
    "gate_structure": "small",
    "gate_review": "small",
}


def role_of(node: str) -> str:
    return NODE_ROLES.get(node, "writer")


def equivalent_model(role: str) -> str:
    from agent.llm import load_config

    return load_config()["profiles"]["balanced"][role][0]


def fetch_prices(models: list[str]) -> dict:
    """{model: {"prompt": usd_per_token, "completion": usd_per_token}} from OpenRouter; saved."""
    response = httpx.get(MODELS_URL, headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"},
                         timeout=60)
    response.raise_for_status()
    listed = {m["id"]: m.get("pricing") or {} for m in response.json()["data"]}
    prices = {m: {"prompt": float(listed[m].get("prompt", 0) or 0), "completion": float(listed[m].get("completion", 0) or 0)}
              for m in models if m in listed}
    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRICES_PATH.write_text(json.dumps({"fetched": time.strftime("%Y-%m-%d %H:%M"), "source": MODELS_URL,
                                       "usd_per_token": prices}, indent=1), encoding="utf-8")
    return prices


def load_prices() -> dict:
    return json.loads(PRICES_PATH.read_text(encoding="utf-8"))["usd_per_token"] if PRICES_PATH.is_file() else {}


def is_free(model: str) -> bool:
    return model.endswith(":free")


def record_costs(record: dict, prices: dict) -> tuple[float, float]:
    """(actual, theoretical) USD of one usage record {node, model, prompt_tokens, completion_tokens, cost}."""
    actual = float(record.get("cost") or 0)
    if not is_free(record["model"]):
        return actual, actual
    price = prices.get(equivalent_model(role_of(record["node"])), {"prompt": 0.0, "completion": 0.0})
    theoretical = record["prompt_tokens"] * price["prompt"] + record["completion_tokens"] * price["completion"]
    return actual, theoretical


def total_costs(records: list[dict], prices: dict) -> tuple[float, float]:
    actual = theoretical = 0.0
    for r in records:
        a, t = record_costs(r, prices)
        actual += a
        theoretical += t
    return round(actual, 6), round(theoretical, 6)
