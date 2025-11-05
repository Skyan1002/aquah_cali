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

The manager uses event-level metrics (default: top three peaks) to determine the
best candidate each round. Hydrograph and per-event figures are generated only
for the top-performing candidate and are fed back into the LLM agents for the
next round.
