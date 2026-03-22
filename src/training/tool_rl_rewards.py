"""
Rule-based rewards for agent RL: reasoning traces, tool XML, valid tool names, JSON params.
"""

from __future__ import annotations

import json
import math
import re

ALLOWED_TOOL_NAMES = frozenset({"web_search", "file_reader"})

_TOOL_NAME_RE = re.compile(r"<tool_name>\s*([^<\s]+)\s*</tool_name>", re.IGNORECASE)
_PARAMS_RE = re.compile(r"<parameters>\s*([\s\S]*?)\s*</parameters>", re.IGNORECASE)


def score_tool_completion(text: str, weights: dict[str, float] | None = None) -> float:
    """
    Higher is better. Designed for the project's Thought / Action / tool_use / Final Answer schema.
    """
    w = {
        "thought": 0.18,
        "action": 0.12,
        "tool_use_pair": 0.22,
        "tool_name_ok": 0.2,
        "json_ok": 0.28,
        "final_answer": 0.12,
        "length_penalty": 0.08,
        "garbage_penalty": 0.15,
    }
    if weights:
        w.update(weights)

    if not text or not text.strip():
        return -1.0

    t = text.strip()
    score = 0.0

    if re.search(r"\bThought:", t, re.IGNORECASE) or re.search(r"Thought:\s*Step", t, re.IGNORECASE):
        score += w["thought"]

    if re.search(r"\bAction:", t, re.IGNORECASE):
        score += w["action"]

    if "<tool_use>" in t and "</tool_use>" in t:
        score += w["tool_use_pair"]

    for m in _TOOL_NAME_RE.finditer(t):
        name = m.group(1).strip()
        if name in ALLOWED_TOOL_NAMES:
            score += w["tool_name_ok"]
            break

    json_hits = 0
    for m in _PARAMS_RE.finditer(t):
        body = m.group(1).strip()
        try:
            obj = json.loads(body)
            if isinstance(obj, dict) and len(obj) > 0:
                json_hits += 1
                score += w["json_ok"]
        except (json.JSONDecodeError, TypeError):
            score -= 0.12

    if json_hits > 1:
        score += 0.05 * (json_hits - 1)

    if re.search(r"Final Answer:", t, re.IGNORECASE):
        score += w["final_answer"]

    ln = len(t)
    if ln > 3500:
        score -= w["length_penalty"] * min(3.0, (ln - 3500) / 1500.0)

    ctrl = sum(1 for c in t if ord(c) < 32 and c not in "\n\r\t")
    if ctrl > 0:
        score -= w["garbage_penalty"] * min(5.0, ctrl / 5.0)

    non_ascii_ratio = sum(1 for c in t if ord(c) > 127) / max(len(t), 1)
    if non_ascii_ratio > 0.35:
        score -= w["garbage_penalty"] * (non_ascii_ratio - 0.35)

    return float(max(-2.0, min(3.0, score)))


def summarize_batch_rewards(rewards: list[float]) -> dict[str, float]:
    if not rewards:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / max(n, 1)
    return {
        "mean": mean,
        "max": max(rewards),
        "min": min(rewards),
        "std": math.sqrt(var),
    }
