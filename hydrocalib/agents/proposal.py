"""First-stage proposal agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .types import RoundContext
from .utils import b64_image, coerce_updates, extract_json_block, get_client
from ..config import LLM_MODEL_DEFAULT, TEMPERATURE_DEFAULT
from ..parameters import ParameterSet


PROPOSAL_SYSTEM_PROMPT = """You are a hydrologic calibration strategist.
Given current metrics and history, propose diverse parameter update strategies.
Return STRICT JSON with a `candidates` list; each candidate needs an `id`,
`goal` (short description) and `updates` mapping parameter names to either
numbers or {"op": "*|+|-|=", "value": number} for multiplicative/additive adjustments.
Keep proposals safe, within hydrologic intuition, and ensure diversity."""


class ProposalAgent:
    def __init__(self,
                 model: str = LLM_MODEL_DEFAULT,
                 temperature: float = TEMPERATURE_DEFAULT):
        self.model = model
        self.temperature = temperature
        self.client = get_client()

    def build_prompt(self, context: RoundContext, k: int) -> str:
        payload = {
            "round": context.round_index,
            "current_params": context.params,
            "aggregate_metrics": context.aggregate_metrics,
            "event_metrics": context.event_metrics,
            "history_summary": context.history_summary,
            "requested_candidates": k,
            "notes": context.description,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def propose(self, context: RoundContext, k: int) -> List[Dict[str, Any]]:
        user_prompt = self.build_prompt(context, k)
        if context.images:
            content = [{"type": "text", "text": user_prompt}]
            content.extend({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(img)}"}} for img in context.images)
            messages = [
                {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        raw = response.choices[0].message.content
        data = extract_json_block(raw)
        candidates = data.get("candidates", [])[:k]
        return candidates

    def apply_candidates(self, base_params: ParameterSet, candidates: List[Dict[str, Any]]) -> List[ParameterSet]:
        param_sets: List[ParameterSet] = []
        for cand in candidates:
            updates = cand.get("updates", {})
            new_params = coerce_updates(base_params, updates)
            param_sets.append(new_params)
        return param_sets


__all__ = ["ProposalAgent"]
