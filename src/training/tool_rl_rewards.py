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
    Higher is better. Designed for the step-by-step Agentic schema.
    A valid completion should EITHER be a tool call (Thought + Action) OR a Final Answer.
    """
    w = {
        "thought": 0.20,
        "action": 0.20,
        "tool_use_pair": 0.20,
        "tool_name_ok": 0.15,
        "json_ok": 0.25,
        "final_answer": 0.80,  # High reward for proper final answer format
        "hallucination_penalty": 0.50, # Penalize generating both tool call AND final answer
        "garbage_penalty": 0.15,
    }
    if weights:
        w.update(weights)

    if not text or not text.strip():
        return -1.0

    t = text.strip()
    score = 0.0

    has_thought = re.search(r"\bThought:", t, re.IGNORECASE) or re.search(r"Thought:\s*Step", t, re.IGNORECASE)
    has_action = re.search(r"\bAction:", t, re.IGNORECASE)
    has_tool_pair = "<tool_use>" in t and "</tool_use>" in t
    has_final_answer = re.search(r"Final Answer:", t, re.IGNORECASE)

    if has_thought:
        score += w["thought"]

    # If it's a tool call step
    if has_action and has_tool_pair:
        score += w["action"]
        score += w["tool_use_pair"]

        # Check valid tool names
        m_name = _TOOL_NAME_RE.search(t)
        if m_name and m_name.group(1).lower() in ALLOWED_TOOL_NAMES:
            score += w["tool_name_ok"]

        # Check JSON validity
        m_params = _PARAMS_RE.findall(t)
        json_hits = 0
        for params_str in m_params:
            try:
                obj = json.loads(params_str.strip())
                if isinstance(obj, dict):
                    json_hits += 1
            except (json.JSONDecodeError, TypeError):
                pass
        
        if json_hits > 0:
            score += w["json_ok"]
        else:
            score -= 0.15

    # If it's a final answer step
    elif has_final_answer:
        score += w["final_answer"]

    # Penalize if it tries to do BOTH (hallucinating observations)
    if has_action and has_final_answer:
        score -= w["hallucination_penalty"]

    # Garbage penalties
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
