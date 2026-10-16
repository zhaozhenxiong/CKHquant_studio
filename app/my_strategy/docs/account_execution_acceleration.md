# Account replay execution acceleration

The causal daily scheduler still owns observation admission, held-position risk,
logical quotas, scoring and the exact sequence of Broker/Ledger updates. Kronos
inference remains in the coordinator with its original model lock, parameters,
seeded RNG context and quality gates. In particular, a queue showing `0/1` means
zero of **today's one stock**, not zero of the entire historical replay. Progress
also reports the actual completed/total sessions and cumulative observations.

`AccountPredictionInputs` reuses each verified raw snapshot already loaded for
the request. Two CPU threads prepare date-truncated copies, exact adapter CSV
`data_version` and full prediction request SHA. At most eight prepared prefixes
are outstanding, with lookahead of two observed dates per requested symbol.
Workers do not call models, stores, scorers or Brokers. The scheduler alone
consumes a prefix; speculative preparation never consumes observation quota.
All pending work is cancelled or drained before leaving the request. Missing
dates remain errors. Prefix metadata is recomputed without future legacy or
unverified-row counts. The immutable verified calendar is fixed for the request.

New service replays pace completed-day cache checkpoints to five seconds.
The Broker still processes every day. Completed, partial, cancelled and deadline
interruption checkpoints always materialize the complete original state and
pending observations. This reduces repeated copies, normalization and gzip of
the ever-growing four-policy history. After an unexpected process crash, up to
five seconds of additional **completed-day work**, plus the currently unfinished
day, may need replay; forecast caches remain available and quotas are accounted
from the last persisted checkpoint. Normal cancellation preserves the exact
pending observation outcomes. Nothing updates an original saved result in place.

`light_account_compatibility.py` contains one reviewed pair of complete exact
code identity dictionaries. Only that previous execution adapter and this exact
accelerated adapter are compatible. Every unchanged model, source, scorer,
policy and Broker dependency must still match. The new helper hash is mandatory;
an added, removed or changed dependency rejects the bridge. The adapter's actual
SHA, pipeline bounds and checkpoint policy are recorded separately in new result
identity. Later adapter changes reject an already frozen accelerated account.
No old account's code, model or configuration is rewritten. Accounts already
incompatible before this execution change remain blocked.

Acceptance uses isolated temporary stores and read-only production raw/CAS input,
the original frozen historical forecasts and scores. It compares complete daily
inventory/equity, Ledger trades, orders, decisions, metrics, state, opportunities
and Agent evidence across all four policies. Component preparation timings and
complete replay timings are separate: fewer database reads or two CPU threads
alone do not establish an end-to-end speedup. No account task is restarted and no
model, sampling count, strategy gate or automatic allocation qualification is
changed by this optimization.
