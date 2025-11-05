"""Two-stage calibration manager orchestrating proposal/evaluation agents."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (DEFAULT_GAUGE_NUM, DEFAULT_PEAK_PICK_KWARGS, DEFAULT_SIM_FOLDER,
                       EVENTS_FOR_AGGREGATE, IMPROVE_PATIENCE, MAX_STEPS_DEFAULT)
from ..history import CandidateRecord, HistoryStore, RoundRecord
from ..metrics import aggregate_event_metrics, compute_event_metrics
from ..parameters import ParameterSet
from ..peak_events import pick_peak_events
from ..plotting import plot_event_windows, plot_hydrograph_with_precipitation
from ..simulation import SimulationResult, SimulationRunner, run_simulations_parallel
from .evaluation import EvaluationAgent
from .proposal import ProposalAgent
from .types import RoundContext


@dataclass
class CandidateOutcome:
    simulation: SimulationResult
    params: ParameterSet
    windows: List[Tuple]
    event_metrics: List[Dict[str, float]]
    aggregate_metrics: Dict[str, float]
    hydrograph_path: Optional[str] = None
    event_figures: List[str] = field(default_factory=list)


class TwoStageCalibrationManager:
    def __init__(self,
                 args_obj,
                 simu_folder: str = DEFAULT_SIM_FOLDER,
                 gauge_num: str = DEFAULT_GAUGE_NUM,
                 n_candidates: int = 8,
                 n_peaks: int = EVENTS_FOR_AGGREGATE,
                 include_max_event_images: int = 3,
                 peak_pick_kwargs: Optional[Dict] = None,
                 history_path: Optional[str] = None):
        self.args_obj = args_obj
        self.current_params = ParameterSet.from_object(args_obj)
        self.runner = SimulationRunner(simu_folder=simu_folder, gauge_num=gauge_num)
        self.proposal_agent = ProposalAgent()
        self.evaluation_agent = EvaluationAgent()
        self.n_candidates = n_candidates
        self.n_peaks = n_peaks
        self.include_max_event_images = include_max_event_images
        self.peak_pick_kwargs = peak_pick_kwargs or DEFAULT_PEAK_PICK_KWARGS
        hist_path = history_path or (Path(simu_folder) / "results" / "calibration_history.json")
        self.history = HistoryStore(Path(hist_path))
        self.best_outcome: Optional[CandidateOutcome] = None
        self.round_index = 0
        self.stall = 0

    def initialize_baseline(self) -> None:
        baseline_result = self.runner.run(self.current_params, round_index=0, candidate_index=0)
        outcome = self._process_result(baseline_result)
        self._generate_plots(outcome)
        self.best_outcome = outcome
        self.history.update_best(outcome.aggregate_metrics, outcome.params.values.copy(), 0, 0)
        self.history.save()

    def _process_result(self, result: SimulationResult) -> CandidateOutcome:
        windows = pick_peak_events(result.csv_path, n=self.n_peaks, **self.peak_pick_kwargs)
        event_metrics = compute_event_metrics(result.csv_path, windows)
        aggregate = aggregate_event_metrics(event_metrics, top_n=self.n_peaks)
        return CandidateOutcome(result, result.params, windows, event_metrics, aggregate)

    def _generate_plots(self, outcome: CandidateOutcome) -> None:
        outcome.hydrograph_path = plot_hydrograph_with_precipitation(outcome.simulation.csv_path, show=False)
        peaks_dir = Path(outcome.simulation.output_dir) / "peaks"
        outcome.event_figures = plot_event_windows(
            outcome.simulation.csv_path,
            outcome.windows,
            out_dir=str(peaks_dir),
        )[:self.include_max_event_images]

    def _history_summary(self, last_k: int = 3) -> str:
        if not self.history.rounds:
            return "No prior rounds."
        tail = self.history.rounds[-last_k:]
        parts = []
        for round_record in tail:
            best = next((c for c in round_record.candidates if c.candidate_index == round_record.best_candidate_index), None)
            if not best:
                continue
            metrics = best.metrics
            parts.append(
                f"r{round_record.round_index}: NSE={metrics.get('NSE', float('nan')):.3f} "
                f"CC={metrics.get('CC', float('nan')):.3f} "
                f"KGE={metrics.get('KGE', float('nan')):.3f}"
            )
        return " | ".join(parts) if parts else "No prior rounds."

    def _build_context(self) -> RoundContext:
        assert self.best_outcome is not None
        description = "Top candidate metrics averaged across selected events."
        images = []
        if self.best_outcome.hydrograph_path:
            images.append(self.best_outcome.hydrograph_path)
        images.extend(self.best_outcome.event_figures[:self.include_max_event_images])
        return RoundContext(
            round_index=self.round_index,
            params=self.best_outcome.params.values.copy(),
            aggregate_metrics=self.best_outcome.aggregate_metrics,
            event_metrics=self.best_outcome.event_metrics[: self.n_peaks],
            history_summary=self._history_summary(),
            description=description,
            images=images,
        )

    def _history_payload(self) -> Dict[str, Any]:
        return json.loads(self.history.path.read_text()) if self.history.path.exists() else {}

    def _select_best(self, outcomes: List[CandidateOutcome]) -> int:
        best_idx = -1
        best_score = -math.inf
        for idx, outcome in enumerate(outcomes):
            nse = outcome.aggregate_metrics.get("NSE", float("nan"))
            score = nse if np.isfinite(nse) else -math.inf
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def run(self, max_rounds: int = MAX_STEPS_DEFAULT) -> None:
        if self.best_outcome is None:
            self.initialize_baseline()
        for r in range(1, max_rounds + 1):
            self.round_index = r
            context = self._build_context()
            proposals = self.proposal_agent.propose(context, self.n_candidates)
            proposal_params = self.proposal_agent.apply_candidates(self.best_outcome.params, proposals)

            refined_candidates, eval_meta = self.evaluation_agent.refine(
                context,
                proposals,
                self._history_payload(),
                self.n_candidates,
            )
            refined_params = self.evaluation_agent.apply_candidates(self.best_outcome.params, refined_candidates)
            if not refined_params:
                refined_candidates = proposals
                refined_params = proposal_params

            results = run_simulations_parallel(self.runner, refined_params, round_index=r)
            outcomes = [self._process_result(res) for res in results]

            best_idx = self._select_best(outcomes)
            best_outcome = outcomes[best_idx]
            self._generate_plots(best_outcome)
            self.best_outcome = best_outcome
            self.current_params = best_outcome.params.copy()
            self.current_params.to_object(self.args_obj)

            candidate_records = [
                CandidateRecord(
                    candidate_index=outcome.simulation.candidate_index,
                    params=outcome.params.values.copy(),
                    metrics=outcome.aggregate_metrics,
                    event_metrics=outcome.event_metrics,
                )
                for outcome in outcomes
            ]
            round_record = RoundRecord(
                round_index=r,
                proposals=proposals,
                refined_candidates=refined_candidates,
                candidates=candidate_records,
                best_candidate_index=best_outcome.simulation.candidate_index,
                rationale=eval_meta.get("rationale", ""),
                risk=eval_meta.get("risk", ""),
                focus=eval_meta.get("focus", ""),
            )
            self.history.rounds.append(round_record)
            self.history.update_best(best_outcome.aggregate_metrics, best_outcome.params.values.copy(), r, best_outcome.simulation.candidate_index)
            self.history.save()

            print(
                f"[Round {r}] Best candidate {best_outcome.simulation.candidate_index}: "
                f"NSE={best_outcome.aggregate_metrics.get('NSE', float('nan')):.3f} "
                f"CC={best_outcome.aggregate_metrics.get('CC', float('nan')):.3f} "
                f"KGE={best_outcome.aggregate_metrics.get('KGE', float('nan')):.3f}"
            )

            if r >= IMPROVE_PATIENCE:
                break


__all__ = ["TwoStageCalibrationManager", "CandidateOutcome"]
