# aquah_cali

This repository provides a modular calibration toolkit for EF5/CREST-based
hydrologic simulations. The core package `hydrocalib` contains utilities to
run simulations, evaluate events, generate figures, and drive a two-stage LLM
calibration workflow that proposes and evaluates multiple parameter sets in
parallel.

## Key modules

- `hydrocalib/config.py` – constants for parameter bounds, defaults, and model
  configuration.
- `hydrocalib/parameters.py` – helpers for managing parameter sets with bounds
  and guarded updates.
- `hydrocalib/simulation.py` – EF5 control-file rendering and multi-threaded
  simulation execution.
- `hydrocalib/metrics.py` – event-level metrics (NSE, CC, KGE, lag) and
  aggregation over the top events.
- `hydrocalib/peak_events.py` – peak detection utilities used to define event
  windows.
- `hydrocalib/plotting.py` – hydrograph and per-event visualization helpers.
- `hydrocalib/history.py` – serialization of calibration rounds and best
  results.
- `hydrocalib/agents` – LLM agents (`ProposalAgent`, `EvaluationAgent`) and the
  `TwoStageCalibrationManager` orchestrating proposal, refinement, and parallel
  simulation of candidate parameter sets.

## Usage outline

1. Instantiate an object holding your CREST/KW parameters (attributes must match
   the names defined in `config.PARAM_BOUNDS`).
2. Create a `TwoStageCalibrationManager`, passing the parameter object and the
   simulation folder that contains the baseline `control.txt`.
3. Call `manager.run(max_rounds=...)` to execute calibration rounds. Each round
   generates candidate parameter sets, evaluates them in parallel, and updates
   the history JSON under `<simu_folder>/results/calibration_history.json`.

The manager aggregates per-event metrics (default: top three peaks) **and** the
full-period NSE/CC/KGE when ranking candidates. Hydrograph and per-event figures
are generated only for the top-performing candidate and are fed back into the
LLM agents for the next round.

### Runtime visibility and artifacts

- Each phase of a round now prints explicit progress logs: proposal requests,
  initial suggestions, post-review refinements, simulation launches, and
  per-candidate metric summaries (including the blended ranking score).
- The best-performing candidate of the entire run is mirrored under
  `<simu_folder>/results/best/` with a `summary.json`, the hydrograph plot, and
  per-event figures for quick inspection. Individual EF5 stdout/stderr logs are
  also captured in `<candidate>/logs/ef5.log`.
- Parallel simulations respect a configurable worker cap (`max_workers` on the
  manager) and default to the available CPU count to avoid resource contention
  when exploring many candidates.
