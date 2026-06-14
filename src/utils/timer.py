"""High-precision wall-clock timer with named event support."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TimerEvent:
    name: str
    start_ns: int
    end_ns: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        if self.end_ns is None:
            return -1.0
        return (self.end_ns - self.start_ns) / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0


class Timer:
    """Per-worker timer that records named events with ns precision."""

    def __init__(self, rank: int):
        self.rank = rank
        self.events: List[TimerEvent] = []
        self._active: Dict[str, TimerEvent] = {}

    def start(self, name: str, **metadata) -> None:
        self._active[name] = TimerEvent(
            name=name, start_ns=time.perf_counter_ns(), metadata=metadata
        )

    def stop(self, name: str) -> TimerEvent:
        ev = self._active.pop(name)
        ev.end_ns = time.perf_counter_ns()
        self.events.append(ev)
        return ev

    def record(self, name: str, duration_ns: int, **metadata) -> None:
        now = time.perf_counter_ns()
        self.events.append(
            TimerEvent(name=name, start_ns=now - duration_ns, end_ns=now, metadata=metadata)
        )

    def summary(self) -> Dict[str, float]:
        """Aggregate durations by event name prefix."""
        agg: Dict[str, float] = {}
        for ev in self.events:
            prefix = ev.name.split("/")[0]
            agg[prefix] = agg.get(prefix, 0.0) + ev.duration_us
        return agg

    def reset(self) -> None:
        self.events.clear()
        self._active.clear()
