"""Second-stage evaluation agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .types import RoundContext
from .utils import b64_image, coerce_updates, extract_json_block, get_client
from ..config import LLM_MODEL_REASONING, TEMPERATURE_DEFAULT
from ..parameters import ParameterSet


EVALUATION_SYSTEM_PROMPT = """You are a hydrologic calibration reviewer.
You receive initial candidate parameter updates and recent history data.
Assess the candidates, adjust them for diversity and safety, and output
STRICT JSON with `refined_candidates`: list of items each containing `id`,
`origin` (which proposal inspired it), `rationale`, and `updates` mapping
parameters to numbers or operation dictionaries. Encourage diverse
strategies (e.g., timing-first, peak-height adjustments, baseflow tuning).
Ensure each refined candidate remains within hydrologic bounds."""


class EvaluationAgent:
    def __init__(self,
                 model: str = LLM_MODEL_REASONING,
                 temperature: float = TEMPERATURE_DEFAULT):
        self.model = model
        self.temperature = temperature
        self.client = get_client()

    def build_prompt(self,
                     context: RoundContext,
                     proposals: List[Dict[str, Any]],
                     history_payload: Dict[str, Any],
                     k: int) -> str:
        payload = {
            "round": context.round_index,
            "current_params": context.params,
            "aggregate_metrics": context.aggregate_metrics,
            "event_metrics": context.event_metrics,
            "history_summary": context.history_summary,
            "history_payload": history_payload,
            "initial_proposals": proposals,
            "requested_candidates": k,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def refine(self,
               context: RoundContext,
               proposals: List[Dict[str, Any]],
               history_payload: Dict[str, Any],
               k: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        user_prompt = self.build_prompt(context, proposals, history_payload, k)
        if context.images:
            content = [{"type": "text", "text": user_prompt}]
            content.extend({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(img)}"}} for img in context.images)
            messages = [
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        raw = response.choices[0].message.content
        data = extract_json_block(raw)
        refined = data.get("refined_candidates", [])[:k]
        meta = {
            "rationale": data.get("rationale", ""),
            "risk": data.get("risk", ""),
            "focus": data.get("focus", ""),
        }
        return refined, meta

    def apply_candidates(self, base_params: ParameterSet, refined: List[Dict[str, Any]]) -> List[ParameterSet]:
        param_sets: List[ParameterSet] = []
        for cand in refined:
            updates = cand.get("updates", {})
            new_params = coerce_updates(base_params, updates)
            param_sets.append(new_params)
        return param_sets


__all__ = ["EvaluationAgent"]
