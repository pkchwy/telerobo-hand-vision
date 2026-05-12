"""Time-sync integration hook (placeholder).

Two-machine time synchronisation is intentionally out of scope for this
refactor; a separate effort handles it. This module is the single point of
contact: replace `get_offset_ns` with whatever your sync layer (chrony,
PTP, NTP query, etc.) reports, and the metrics CSV will pick it up
automatically.

Convention:
    - role: free-form string ("client", "server", or whatever the caller
      passes). Useful if the implementation needs to know which side it
      runs on.
    - return value: signed offset in nanoseconds that should be ADDED to
      this machine's `time.time_ns()` value to map it onto the shared
      reference clock. Positive => this machine's clock is behind.

Default returns 0 so latency math degrades to "single-machine wall clock".
"""

from __future__ import annotations


def get_offset_ns(role: str = "client") -> int:
    return 0


def time_sync_method() -> str:
    """Identifier written into run_metadata.json. Update when implemented."""
    return "none"
