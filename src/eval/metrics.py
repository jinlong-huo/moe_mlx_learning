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


# ── OCS-specific metrics ──────────────────────────────────────────────────


def ocs_overlap_ratio(events: List[TimerEvent]) -> float:
    """Measure what fraction of OCS pre-establishment time was hidden behind compute.

    Looks for ocs_pre_establish events and checks their time overlap with
    any compute events. A ratio near 1.0 means circuit reconfig is fully
    hidden; near 0.0 means it's fully exposed on the critical path.
    """
    ocs_events = [e for e in events if "ocs_pre_establish" in e.name]
    compute_events = [e for e in events if "compute" in e.name]

    if not ocs_events:
        return 1.0  # no OCS activity = nothing exposed

    total_ocs_ns = sum(e.duration_us for e in ocs_events) * 1000
    overlapped_ocs_ns = 0.0

    for oe in ocs_events:
        oe_start_ns = oe.start_ns
        oe_end_ns = oe.end_ns

        for ce in compute_events:
            ce_start_ns = ce.start_ns
            ce_end_ns = ce.end_ns

            overlap_start = max(oe_start_ns, ce_start_ns)
            overlap_end = min(oe_end_ns, ce_end_ns)
            if overlap_start < overlap_end:
                overlapped_ocs_ns += overlap_end - overlap_start

    if total_ocs_ns == 0:
        return 1.0
    return overlapped_ocs_ns / total_ocs_ns


def ocs_step_metrics(events: List[TimerEvent]) -> dict:
    """Extend step_metrics with OCS-specific fields."""
    base = step_metrics(events)

    ocs_pre_estab_events = [e for e in events if "ocs_pre_establish" in e.name]
    base["ocs_pre_establish_us"] = sum(e.duration_us for e in ocs_pre_estab_events)
    base["ocs_pre_establish_count"] = len(ocs_pre_estab_events)
    base["ocs_overlap_ratio"] = ocs_overlap_ratio(events)

    # Compute effective overlap: include OCS reconfig as "comm" cost
    comm_us = base["comm_us"] + base["ocs_pre_establish_us"]
    compute_us = base["compute_us"]
    total = comm_us + compute_us
    base["effective_comm_pct"] = (comm_us / total * 100) if total > 0 else 0

    return base
