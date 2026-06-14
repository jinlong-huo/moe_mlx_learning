"""Profiler: aggregate traces across all ranks and produce
summary statistics plus Chrome Trace JSON for visualization.

Usage:
  profiler = Profiler(trace_dir="outputs/traces")
  profiler.collect()        # read all rank_*_trace.json files
  profiler.summary()         # print per-rank stats
  profiler.merge_traces()    # write merged Chrome Trace
"""
from __future__ import annotations

import json
import os
import glob
from typing import List, Dict

from src.utils.timer import TimerEvent


class Profiler:
    """Aggregate and analyze timeline traces from multiple ranks."""

    def __init__(self, trace_dir: str = "outputs/traces"):
        self.trace_dir = trace_dir
        self.rank_events: Dict[int, List[dict]] = {}

    def collect(self) -> int:
        """Load all rank trace files from disk. Returns number of ranks found."""
        self.rank_events.clear()
        pattern = os.path.join(self.trace_dir, "rank_*_trace.json")
        for fpath in sorted(glob.glob(pattern)):
            basename = os.path.basename(fpath)
            # Extract rank number: "rank_00_trace.json" → 0
            rank_str = basename.split("_")[1]
            rank = int(rank_str)
            with open(fpath) as f:
                data = json.load(f)
                self.rank_events[rank] = data.get("traceEvents", [])
        return len(self.rank_events)

    def summary(self) -> dict:
        """Compute per-rank and global statistics."""
        rank_stats = {}
        for rank, events in self.rank_events.items():
            comm_events = [e for e in events if e["cat"] == "comm"]
            compute_events = [e for e in events if e["cat"] == "compute"]
            comm_dur = sum(e["dur"] for e in comm_events)
            compute_dur = sum(e["dur"] for e in compute_events)
            total_dur = comm_dur + compute_dur
            rank_stats[rank] = {
                "comm_us": comm_dur,
                "compute_us": compute_dur,
                "total_us": total_dur,
                "comm_pct": (comm_dur / total_dur * 100) if total_dur > 0 else 0,
            }
        return rank_stats

    def merge_traces(self, output_path: str = "outputs/traces/merged_trace.json") -> str:
        """Merge all rank traces into a single Chrome Trace file.
        Each rank gets a unique PID, making them separate rows in chrome://tracing.
        """
        all_events = []
        for rank, events in self.rank_events.items():
            # Re-tag pid so each rank is a separate row
            for ev in events:
                ev["pid"] = rank
                ev["tid"] = 0
            all_events.extend(events)

        payload = {"traceEvents": all_events, "displayTimeUnit": "ns"}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        return os.path.abspath(output_path)
