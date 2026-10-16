# Kronos loading and factual execution evidence

Kronos-base and its tokenizer still use the pinned official assets and unchanged
frozen vendor implementation. The adapter constructs the main model on the
PyTorch meta device and strictly assigns every parameter and persistent buffer
from the original safetensors file. This avoids allocating and randomly
initializing an unused full-size parameter set before replacing it with the
pinned weights. Any missing state key or remaining meta parameter/buffer fails
before inference. The tokenizer keeps its original CPU construction so its
nonpersistent group codebook is fully initialized. CPU construction still runs
inside the original RNG fork; inference keeps the original CPU/MPS/CUDA RNG
save, seed and restore context.

There is no sampler substitution, precision change, parameter reduction, quality
repair or sample reduction. A tested `masked_fill_` sampler candidate showed no
material speed gain and was discarded. Both CPU and MPS still invoke the frozen
vendor inference function, with the same sample axis and RNG call order.

Every new prediction reports `resource.execution` with schema
`kronos_execution_v1`: the requested and resolved devices, observed model and
tokenizer parameter devices, observed input tensor devices,
`tensor_devices_verified`, `fresh_inference`, `inference_calls` and
`synchronization_completed`. All three observed device sets must equal the
resolved device. PyTorch's single Apple tensor address `mps:0` is normalized to
`mps`; CUDA retains its ordinal, including resolving a bare requested `cuda` to
the actual current CUDA ordinal. Explicit unavailable devices still raise.
Actual inference is synchronized before declaring the execution complete. These
facts establish the device of this actual call; they are not a GPU utilization
percentage. A stored result's old execution record is provenance only. Callers
must exclude `reused == true` results when counting this task's fresh GPU calls.

`resource.timings` separately reports input preparation, pinned asset checking,
vendor/source checking, model-lock waiting, model loading, input transfer,
synchronized inference and summary processing, in seconds. Device checking and
synchronization do not enter forecast numerical output. The adapter's actual
source SHA changes the request identity, so old prediction-cache identities and
in-flight execution checkpoints are not silently adopted. Existing saved model,
account, raw input and result objects are not rewritten.

Performance evidence distinguishes a fresh Python process with an unloaded model
from a warm in-process model. Both may share the operating system file and MPS
kernel caches. The prediction probe starts after the Torch import and raw-frame
loading; it does not measure a whole API request or account replay. An initial
unprimed MPS call includes extra kernel setup and is recorded separately from
interleaved fresh-process before/after measurements. It must not be used to
attribute all first-call improvement to model loading. Warm inference is not
claimed faster. Full-account speed must be measured separately with the actual
causal scheduler and Broker, rather than inferred from this component probe.
