# Native preparation acceleration and database reuse

The user authorized cancellation of the active training task, complete execution
of the preparation acceleration plan, validation of equivalent results before
switching, and database storage of reusable structure outputs.

The cancelled task is `6aa66e7700d247bd871e53cdacbeaafd`. Its complete 3194-stock raw
snapshot ending 2026-10-12 remains frozen. Its partial native output is preserved
as diagnostic evidence and is never relabelled as a new-source checkpoint.

## Decision

Use bounded continuous stock preparation, atomic stock checkpoints, measured
stage progress and a verified SQLite cache. Reuse native inputs only under the
same raw observations, liquidity membership, verified calendar, native input
contract, actual engine, schemas and generating source. Native input identity is
independent of training epochs, research capital and scheduling resources.
Changing native parameters or observations produces a cache miss. New trading
days are rebuilt; incremental engine-state replay has not been introduced.

Retain only the latest two bars per frequency in the aggregation buffer, while
keeping complete closed history and native analysis state. Share per-time-point
scalar pen/zone reads and strictly verified compatible Wyckoff preparation.
The engine's Rust algorithm, source manifest and installed binary are unchanged.

Preparation resume uses a coordinator OS lock and a short commit lock. Workers
compute in attempt-specific locations and must prove current attempt and live
parent ownership before publishing immutable outputs and checkpoints. An old
worker cannot publish after a new attempt has taken ownership.

Local Git integration exposed a simultaneous first-open WAL transition race.
The cache opener retries only SQLite BUSY/LOCKED errors within its connection
wait budget and requires the actual journal mode to be WAL. Other database,
identity and content errors remain explicit failures. This storage-only repair
does not change native calculations or outputs. The active training source
remains frozen; integration does not edit its generating files.

## Actual validation evidence

- Six real stocks and 10,905 observed daily rows: every aggregate, closed input,
  confirmation, chart payload, signal, native Position decision/operation and
  selected causal prefix is exactly equal.
- Sixty frozen stocks spanning 20 to 2858 observed rows: numerical Parquet and
  native gzip bytes are exactly equal for the original implementation and the
  optimized 6-, 8- and 10-process preparations.
- Fixed-input throughput: original 6 processes 149.459 seconds; optimized 6
  processes 129.688; 8 processes 104.890; 10 processes 92.775. Ten processes reduce
  wall time by 37.9%, with approximately 4 GB peak process-family RSS and no
  observed added swap. These figures describe the representative preparation
  benchmark, not the entire training pipeline.
- A separate 60-stock producer under the final frozen generation hashes supplies
  the cache proof. SQLite insertion and cold lookup/materialization preserve both
  physical output hashes, and cold reading never recalculates the structure.
- Cache, source checkpoints, resume locks, ownership, source validation and UI
  checks cover corruption, wrong prices, incompatible policies, wrong calendars,
  code drift, failed stocks, malformed inputs and concurrent restoration.

Evidence lives in `knowledge_base_system/evidence/20261014-123953-d6ef73c7/`.
`real-native-equivalence.json`, `real-cache-equivalence.json`, the full regression
logs and the deployment manifest are the authoritative validation records.

## Runtime switch

The optimized task uses a new frozen plan and all 3194 stocks, reuses only the
verified complete raw snapshot, and runs 10 CPU preparation processes. Neural
training continues to request MPS. The default SQLite location is
`my_strategy/data/processed/czsc/native_structure.db`.

Chronological separation, mature actual Broker targets, fees, 8% and 2R, risk
conditions and automatic selection qualifications retain their original values.
Performance validation grants no model qualification. Independent certification
remains pending and model scoring waits for qualified, available new heads.
