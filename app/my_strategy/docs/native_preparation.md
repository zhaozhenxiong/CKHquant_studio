# Native preparation, checkpoints and SQLite reuse

Preparation preserves the complete frozen historical liquidity-qualified union.
`stock_batch_size` bounds preparation work; it never selects fewer stocks. The
coordinator keeps at most `min(2 * cpu_workers, stock_batch_size)` tasks in flight,
replenishes work as stocks finish, and writes final records in frozen stock order.

The aggregator retains only its two latest bars per frequency. Every closed bar
and the full analysis state remain in `CzscSession.closed` and `kas`. Period
closure, confirmations, native third-sell geometry and chronological inputs stay
unchanged. Each time point shares its pen and zone scalar reads. Compatible
Wyckoff preparation is reused only after verifying its full inputs and outputs.

## Progress and timing

Each successful stock has an atomic checkpoint. Only validated physical outputs
count as completed. `SOURCE_BATCH` reports completed, failed, in-flight, restored,
database hits and measured stocks per minute. Its ETA covers the current native
stage only. Feature construction, native evidence, Parquet, gzip and verification
timings are retained in checkpoints and the completed source report. A partial
directory or a running process alone does not establish completion.

## Database cache

The default cache is `DATA_ROOT/processed/czsc/native_structure.db`. It stores
numerical feature Parquet and native gzip blobs, their hashes, schema and causal
input identities in SQLite with WAL and atomic transactions. Only the parent
coordinator writes the database; preparation workers compute independent files.
Each run receives its own immutable frozen files directly from verified blobs.
No model weights, targets or labels are imported through the cache.

The key binds raw bytes and observation content, units, liquidity membership,
verified calendar, configuration, schemas, actual engine and generating source
hashes. It excludes physical paths, run IDs and worker counts. Identical inputs
under a new run reuse the stored result. Changing the raw observations, adding a
trading day, changing the calendar, a structural parameter or generating source
produces a miss. This implementation does not incrementally append new days.
Existing corrupted entries fail explicitly; they are never quietly used or
overwritten. Blob reads also check the underlying observation content and dates.

## Safe resume

`reports/native_checkpoint.json` freezes the run's inputs, generation identity
and resource parameters before work starts. Per-stock completed and failed
checkpoints retain both output hashes and timing evidence. A per-run OS lock
prevents overlapping preparation coordinators. Resume checks the frozen raw
source, calendar, configuration, schema, code and every completed output before
skipping it. Changed identities require a new source run. Old partial runs from
before this checkpoint format cannot be resumed as new-format sources.

Use the normal task cancellation API to stop the whole process tree. The UI's
"continue structure preparation and training" action cold-verifies checkpoints
and submits a new frozen training plan. The CLI equivalent is:

```sh
python -m my_strategy.cli train-trend --end YYYY-MM-DD \
  --raw-source-run-id RAW_RUN --resume-native-run-id NATIVE_RUN \
  --cpu-workers SAME_WORKERS --stock-batch-size SAME_BATCH \
  --config SAME_CONFIGURATION --device SAME_DEVICE
```

Full frozen raw inputs remain reusable independently of a native source. All
training, chronological splits, mature Broker labels, fees, the 8% and 2R
conditions and model selection qualifications keep their existing contracts.
