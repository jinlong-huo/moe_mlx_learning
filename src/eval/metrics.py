"""Compute-overlap metrics: measure what fraction of communication
was hidden behind useful computation.

Key metrics:
  - overlap_ratio: time where comm overlapped with compute / total comm time
  - total_wall_time: end-to-end wall clock per step
  - comm_pct: communication time as fraction of total
"""
from __future__ import annotations

from typing import List

from src.utils.timer import TimerEvent


def compute_overlap_ratio(events: List[TimerEvent]) -> float:
    """Estimate the fraction of communication time that overlapped with compute.

    Uses a simple interval-overlap algorithm: for each comm event,
    check if any compute event was active during the same time window.
    """
    comm_events = [e for e in events if e.name.startswith("comm/")]
    compute_events = [e for e in events if e.name.startswith("compute/")]

    if not comm_events:
        return 0.0

    total_comm_ns = sum(e.duration_us for e in comm_events) * 1000
    overlapped_comm_ns = 0.0

    for ce in comm_events:
        ce_start = ce.start_ns
        ce_end = ce.end_ns

        # Check overlap with any compute event
        for xe in compute_events:
            xe_start = xe.start_ns
            xe_end = xe.end_ns

            overlap_start = max(ce_start, xe_start)
            overlap_end = min(ce_end, xe_end)

            if overlap_start < overlap_end:
                overlapped_comm_ns += overlap_end - overlap_start

    if total_comm_ns == 0:
        return 0.0
    return overlapped_comm_ns / total_comm_ns


def step_metrics(events: List[TimerEvent]) -> dict:
    """Aggregate metrics for a single step from its events."""
    if not events:
        return {}

    comm_us = sum(e.duration_us for e in events if "comm" in e.name or "scatter" in e.name or "gather" in e.name)
    compute_us = sum(e.duration_us for e in events if "compute" in e.name)
    route_us = sum(e.duration_us for e in events if "route" in e.name)
    total_us = sum(e.duration_us for e in events)
    overlap = compute_overlap_ratio(events)

    return {
        "total_us": total_us,
        "comm_us": comm_us,
        "compute_us": compute_us,
        "route_us": route_us,
        "overlap_ratio": overlap,
        "num_events": len(events),
    }
