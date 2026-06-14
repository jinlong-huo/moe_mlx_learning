"""Plotting utilities for overlap analysis.

Produces:
  - Timeline Gantt charts per rank
  - Overlap ratio vs micro-batch size
  - Comm/Compute breakdown pie charts

Requires: matplotlib (optional — graceful degradation if not installed)
"""
from __future__ import annotations

from typing import List, Dict


def plot_timeline(events: List[dict], title: str = "MoE Timeline", save_path: str | None = None):
    """Plot a Gantt-style timeline of comm and compute events.

    Falls back to a text summary if matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed — skipping timeline plot")
        _text_timeline(events)
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"comm": "#E74C3C", "compute": "#2ECC71", "route": "#3498DB"}
    y_labels = []
    y_pos = 0

    for ev in events:
        cat = ev.get("cat", "other")
        color = colors.get(cat, "#95A5A6")
        start_ms = ev["ts"] / 1000.0  # us → ms
        dur_ms = ev["dur"] / 1000.0

        ax.barh(y_pos, dur_ms, left=start_ms, height=0.6, color=color, edgecolor="white")
        y_labels.append(f"{ev['name']}")
        y_pos += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def _text_timeline(events: List[dict]) -> None:
    """Text-based timeline fallback."""
    print("\n── Timeline (text) ──")
    for ev in sorted(events, key=lambda e: e["ts"]):
        cat = ev.get("cat", "?")
        name = ev["name"]
        dur_us = ev["dur"]
        ts_us = ev["ts"]
        bar = "█" * min(int(dur_us / 100), 80)
        print(f"  [{cat:8s}] {name:40s} @ {ts_us:10d}us  {bar} ({dur_us:.0f}us)")
    print()


def plot_overlap_curve(
    micro_batch_sizes: List[int],
    overlap_ratios: List[float],
    comm_delays: List[float] | None = None,
    save_path: str | None = None,
):
    """Plot overlap ratio vs micro-batch size, optionally with multiple delay values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed — printing overlap curve as text")
        print("  micro_batch_size  overlap_ratio")
        for mb, ratio in zip(micro_batch_sizes, overlap_ratios):
            print(f"  {mb:16d}  {ratio:.3f}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(micro_batch_sizes, overlap_ratios, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Micro-batch Size")
    ax.set_ylabel("Overlap Ratio")
    ax.set_title("Communication-Computation Overlap")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    if comm_delays:
        for mb, ratio, delay in zip(micro_batch_sizes, overlap_ratios, comm_delays):
            ax.annotate(f"{delay:.0f}us", (mb, ratio), textcoords="offset points",
                       xytext=(0, 10), fontsize=8, ha="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
