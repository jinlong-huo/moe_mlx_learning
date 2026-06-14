"""Timeline export: produce Chrome Trace / Perfetto JSON from timer events.

Output format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
"""
from __future__ import annotations

import json
import os
from typing import List

from src.utils.timer import TimerEvent


def _event_to_trace(ev: TimerEvent, pid: int, tid: int) -> dict:
    return {
        "name": ev.name,
        "cat": ev.name.split("/")[0],
        "ph": "X",
        "ts": ev.start_ns // 1000,   # ns → us for Chrome trace
        "dur": max(0, (ev.end_ns - ev.start_ns)) // 1000,
        "pid": pid,
        "tid": tid,
        "args": ev.metadata,
    }


def export_chrome_trace(
    events: List[TimerEvent],
    filepath: str,
    pid: int = 0,
    tid: int = 0,
    metadata: dict | None = None,
) -> str:
    """Export a list of TimerEvents as a Chrome Trace JSON file.

    Args:
        events: list of completed TimerEvents
        filepath: output .json path
        pid: process id for trace visualization
        tid: thread id for trace visualization
        metadata: optional EP/topology metadata for the viewer

    Returns:
        the absolute path written
    """
    trace_events = [_event_to_trace(ev, pid, tid) for ev in events if ev.end_ns is not None]
    payload: dict = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",
    }
    if metadata:
        payload["_metadata"] = metadata
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(filepath)
