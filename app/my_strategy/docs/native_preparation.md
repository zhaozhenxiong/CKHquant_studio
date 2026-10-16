# Native preparation, checkpoints and SQLite reuse

## Current SH/SZ board summaries

The current board display uses the incremental source cache, independently of
the historical training preparation described below. Its full population is the
verified `sh_sz_a_v1` board projection, including ChiNext and STAR members.
After updating and verifying the requested price date, preview a new full task:

```sh
./khquant-native.sh fill-board-native preview --checkpoint /absolute/path/task.json --price-date YYYY-MM-DD --cpu-workers 6 --batch-size 100
./khquant-native.sh fill-board-native execute --checkpoint /absolute/path/task.json --cpu-workers 6 --budget-seconds 1800
```

Each batch contains at most 100 stocks. Preview defaults to four workers; the
examples explicitly select six. `--cpu-workers` permits 1..12 for a measured
machine with enough CPU and memory. It does not change numerical source identity.
Changing coordinator code still requires a new preview; that preview verifies and
reuses completed immutable source objects instead of trusting an old checkpoint.
Each worker opens its own source-cache connection. Completed summaries have
atomic files and SHA-bound checkpoint records. A partial attempt does not
publish a board generation; an explicit subsequent `execute` revalidates the
same checkpoint before continuing. A cancelled task is terminal and requires a
new preview. The dispatch budget stops new work; in-flight work and final
verification can extend beyond it.

Publication occurs once, after the entire frozen population and actual source
objects are reverified. With more than one worker the existing process pool also
runs the final unchanged `inspect` and `summary` checks, at most one stock per
worker. The parent reads each SHA-bound stock checkpoint and compares the actual
summary, row hash, cache key and frozen input identity before collecting results
in frozen population order. One-worker execution performs these checks serially.
Any final failure or cancellation blocks publication; leases cover verification
as well as preparation. Final verification progress is reported separately and
remains outside the dispatch budget. Input, calendar, producer or executor changes reject
reuse. Missing local history and missing target-day prices remain explicit
unavailable summaries in the original denominator. Final verification independently
reads actual raw history before accepting either missing reason; a re-signed
checkpoint action cannot conceal an existing target-day price. Malformed data and unknown
errors remain failures. `full_published` denotes complete population processing,
not universal eligible structure coverage. Per-board minimum-member and coverage
gates still apply. This task never trains models or resumes cancelled legacy jobs.

## Historical training preparation

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
