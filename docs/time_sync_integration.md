# Time sync integration point

Two-machine clock synchronisation is **not** implemented in this repo on
purpose; a separate effort handles it. This document describes the
single integration surface so that work can plug in without touching the
rest of the pipeline.

## Why we left it as a hook

- Decoupling: vision / network / robot / metrics evolve on their own.
- Honesty: every CSV row carries a `clock_offset_ns` column, so once a
  real sync mechanism returns a non-zero value the data is correctly
  re-anchored without any column-shape change.
- Reversibility: drop in chrony, NTP, PTP, custom UDP ping-pong, etc.;
  the rest of the code does not care which it is.

## The hook

File: [`src/metrics/time_sync_hook.py`](../src/metrics/time_sync_hook.py)

Two functions:

```python
def get_offset_ns(role: str = "client") -> int: ...
def time_sync_method() -> str: ...
```

- `get_offset_ns(role)` is called once per frame from the runner. It
  must return the signed nanosecond offset to ADD to this machine's
  `time.time_ns()` to map it onto the shared reference clock. Positive
  means this machine's clock is behind the reference. Default: `0`.
- `time_sync_method()` returns a short string (e.g. `"chrony"`,
  `"ptp"`, `"ntp"`, `"udp-pingpong"`). Written into
  `run_metadata.json::time_sync_method` so analyses can filter or
  caveat results.

## Where the value lands

- CSV: `clock_offset_ns` column on every frame.
- Metadata: `run_metadata.json::time_sync_method`.

The runner does not interpret the offset — it only records it. Any
post-hoc analysis that needs cross-machine timestamps subtracts the
offset itself, or filters by `time_sync_method != "none"` to exclude
runs without a real sync.

## Suggested implementation pattern

```python
# src/metrics/time_sync_hook.py
from your_sync_module import current_offset, METHOD_NAME

def get_offset_ns(role: str = "client") -> int:
    return current_offset(role)

def time_sync_method() -> str:
    return METHOD_NAME
```

Whatever your sync module does (queries chrony's tracking output, talks
to a PTP daemon, runs its own UDP probes...) lives in `your_sync_module`
and should expose just those two pieces. Keep it cheap: `get_offset_ns`
is called per frame.

## Things this hook does NOT do

- It does not synchronise the clocks. That is the job of the underlying
  daemon / mechanism.
- It does not gate the run. Runs proceed even when offset is `0`.
- It does not log uncertainty by itself. If you want
  offset / rms / root_delay / root_dispersion separately, extend the
  hook to return a small dataclass and add columns to `FrameRecord` /
  `RunMetadata`. The current single-int interface is intentional
  minimum viable surface.
