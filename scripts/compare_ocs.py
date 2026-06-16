#!/usr/bin/env python3
"""Compare identical workloads with OCS disabled vs. enabled.

Runs the same experiment twice — once without OCS (baseline), once with
OCS pipeline mode — and produces a side-by-side comparison report.

The comparison isolates the effect of OCS circuit dynamics on:
  - Total wall time per step
  - Communication overhead (comm as % of total)
  - OCS-specific: circuit reuse ratio, reconfig time, effective overlap

Usage:
  python scripts/compare_ocs.py
  python scripts/compare_ocs.py --trace-dir outputs/traces/ocs_comparison
  python scripts/compare_ocs.py --steps 10 --microbatches 4

Prints a JSON report to stdout and writes a detailed HTML report to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def run_experiment(config_path: str, trace_dir: str) -> bool:
    """Run one experiment config. Returns True on success."""
    cmd = [
        sys.executable, "-m", "src.main",
        "--config", config_path,
        "--trace-dir", trace_dir,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
            print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 120s", file=sys.stderr)
        return False


def load_trace_metrics(trace_dir: str, rank: int = 0) -> dict:
    """Load per-rank trace and extract key metrics."""
    path = os.path.join(trace_dir, f"rank_{rank:02d}_trace.json")
    if not os.path.exists(path):
        return {}

    with open(path) as f:
        data = json.load(f)

    meta = data.get("_metadata", {})
    events = data.get("traceEvents", [])

    # Compute timing metrics from events
    comm_events = [e for e in events if e.get("cat") == "comm"]
    compute_events = [e for e in events if e.get("cat") == "compute"]
    ocs_events = [e for e in events if e.get("cat") == "ocs" or "ocs" in e.get("name", "").lower()]

    comm_dur = sum(e.get("dur", 0) for e in comm_events)
    compute_dur = sum(e.get("dur", 0) for e in compute_events)
    ocs_dur = sum(e.get("dur", 0) for e in ocs_events)
    total_dur = comm_dur + compute_dur + ocs_dur

    ocs_metrics = meta.get("ocs", {}).get("metrics", {})

    return {
        "mode": meta.get("mode", "unknown"),
        "world_size": meta.get("world_size"),
        "num_experts": meta.get("num_experts"),
        "comm_us": comm_dur,
        "compute_us": compute_dur,
        "ocs_us": ocs_dur,
        "total_us": total_dur,
        "comm_pct": (comm_dur / total_dur * 100) if total_dur > 0 else 0,
        "num_events": len(events),
        "ocs": ocs_metrics,
    }


def build_comparison(baseline: dict, ocs: dict) -> dict:
    """Build side-by-side comparison dict."""
    comparison = {
        "baseline": baseline,
        "ocs_enabled": ocs,
    }

    # Compute deltas
    if baseline.get("total_us", 0) > 0 and ocs.get("total_us", 0) > 0:
        baseline_total = baseline["total_us"]
        ocs_total = ocs["total_us"]
        comparison["delta"] = {
            "total_us_absolute": ocs_total - baseline_total,
            "total_us_pct": ((ocs_total - baseline_total) / baseline_total) * 100,
            "comm_us_absolute": ocs.get("comm_us", 0) - baseline.get("comm_us", 0),
            "ocs_us": ocs.get("ocs_us", 0),
        }

    # OCS-specific summary
    ocs_metrics = ocs.get("ocs", {})
    if ocs_metrics:
        total_req = max(ocs_metrics.get("total_requests", 1), 1)
        comparison["ocs_summary"] = {
            "circuit_reuses": ocs_metrics.get("circuit_reuses", 0),
            "circuit_establishes": ocs_metrics.get("circuit_establishes", 0),
            "circuit_evictions": ocs_metrics.get("circuit_evictions", 0),
            "total_reconfig_time_us": ocs_metrics.get("total_reconfig_time_us", 0),
            "reuse_ratio": ocs_metrics.get("reuse_ratio", 0),
            "active_circuits": ocs_metrics.get("active_circuits", 0),
            "max_circuits": ocs_metrics.get("max_circuits", 0),
        }

    return comparison


def build_html_report(comparison: dict) -> str:
    """Generate a self-contained HTML comparison report."""
    baseline = comparison["baseline"]
    ocs = comparison["ocs_enabled"]
    delta = comparison.get("delta", {})
    ocs_summary = comparison.get("ocs_summary", {})

    def _color(val, good_low=True):
        """Green if better, red if worse."""
        if val is None:
            return "#8b949e"
        if good_low:
            return "#3fb950" if val < 0 else "#f85149" if val > 0 else "#8b949e"
        return "#3fb950" if val > 0 else "#f85149" if val < 0 else "#8b949e"

    reuse = ocs_summary.get("reuse_ratio", 0)
    reuse_color = "#3fb950" if reuse > 0.7 else "#d29922" if reuse > 0.3 else "#f85149"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCS vs EPS Comparison Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  h2 {{ color: #f0f6fc; font-size: 18px; margin: 20px 0 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 24px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .card h3 {{ font-size: 15px; margin-bottom: 12px; }}
  .metric {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 14px; }}
  .metric .label {{ color: #8b949e; }}
  .metric .value {{ font-weight: 600; }}
  .delta-row {{ display: flex; justify-content: space-between; padding: 8px 12px;
               background: #0d1117; border-radius: 6px; margin: 4px 0; font-size: 14px; }}
  .pos {{ color: #f85149; }} .neg {{ color: #3fb950; }}
  .reuse-bar {{ background: #21262d; border-radius: 4px; height: 12px; overflow: hidden; margin: 4px 0; }}
  .reuse-fill {{ height: 100%; border-radius: 4px; }}
  pre {{ background: #0d1117; padding: 12px; border-radius: 6px; font-size: 12px; overflow-x: auto; }}
</style>
</head>
<body>

<h1>🔬 OCS vs EPS Comparison Report</h1>
<div class="subtitle">
  Baseline: {baseline.get('mode', '?')} &nbsp;|&nbsp;
  OCS: {ocs.get('mode', '?')} &nbsp;|&nbsp;
  World size: {baseline.get('world_size', '?')} &nbsp;|&nbsp;
  Experts: {baseline.get('num_experts', '?')}
</div>

<h2>⏱ Timing Comparison</h2>
<div class="grid-2">
  <div class="card">
    <h3>EPS Baseline (no OCS)</h3>
    <div class="metric"><span class="label">Total time</span><span class="value">{baseline.get('total_us', 0):.0f} µs</span></div>
    <div class="metric"><span class="label">Comm time</span><span class="value">{baseline.get('comm_us', 0):.0f} µs</span></div>
    <div class="metric"><span class="label">Compute time</span><span class="value">{baseline.get('compute_us', 0):.0f} µs</span></div>
    <div class="metric"><span class="label">Comm %</span><span class="value">{baseline.get('comm_pct', 0):.1f}%</span></div>
    <div class="metric"><span class="label">Events</span><span class="value">{baseline.get('num_events', 0)}</span></div>
  </div>
  <div class="card">
    <h3>OCS Pipeline</h3>
    <div class="metric"><span class="label">Total time</span><span class="value">{ocs.get('total_us', 0):.0f} µs</span></div>
    <div class="metric"><span class="label">Comm time</span><span class="value">{ocs.get('comm_us', 0):.0f} µs</span></div>
    <div class="metric"><span class="label">Compute time</span><span class="value">{ocs.get('compute_us', 0):.0f} µs</span></div>
    <div class="metric"><span class="label">OCS-specific time</span><span class="value">{ocs.get('ocs_us', 0):.0f} µs</span></div>
    <div class="metric"><span class="label">Comm %</span><span class="value">{ocs.get('comm_pct', 0):.1f}%</span></div>
    <div class="metric"><span class="label">Events</span><span class="value">{ocs.get('num_events', 0)}</span></div>
  </div>
</div>

<h2>📊 Delta (OCS − Baseline)</h2>
<div class="card">
  <div class="delta-row">
    <span>Total time</span>
    <span class="{'pos' if delta.get('total_us_absolute', 0) > 0 else 'neg'}">
      {delta.get('total_us_absolute', 0):+.0f} µs ({delta.get('total_us_pct', 0):+.1f}%)
    </span>
  </div>
  <div class="delta-row">
    <span>Comm time</span>
    <span class="{'pos' if delta.get('comm_us_absolute', 0) > 0 else 'neg'}">
      {delta.get('comm_us_absolute', 0):+.0f} µs
    </span>
  </div>
  <div class="delta-row">
    <span>OCS overhead</span>
    <span>{delta.get('ocs_us', 0):.0f} µs</span>
  </div>
  <div style="margin-top:12px;font-size:13px;color:#8b949e;">
    {'⚠ OCS adds overhead — circuits are cold on first use' if delta.get('total_us_absolute', 0) > 0 else '✅ OCS matches or beats baseline — circuits are hot'}
  </div>
</div>

<h2>🔌 OCS Circuit Stats</h2>
<div class="card">
  <div class="metric"><span class="label">Circuit reuses</span><span class="value" style="color:#3fb950">{ocs_summary.get('circuit_reuses', 0)}</span></div>
  <div class="metric"><span class="label">New establishments</span><span class="value" style="color:#d29922">{ocs_summary.get('circuit_establishes', 0)}</span></div>
  <div class="metric"><span class="label">LRU evictions</span><span class="value" style="color:#f85149">{ocs_summary.get('circuit_evictions', 0)}</span></div>
  <div class="metric"><span class="label">Total reconfig time</span><span class="value">{ocs_summary.get('total_reconfig_time_us', 0):.0f} µs</span></div>
  <div class="metric"><span class="label">Active circuits</span><span class="value">{ocs_summary.get('active_circuits', 0)} / {ocs_summary.get('max_circuits', '?')}</span></div>
  <div class="metric">
    <span class="label">Reuse ratio</span>
    <span class="value" style="color:{reuse_color}">{(reuse * 100):.1f}%</span>
  </div>
  <div class="reuse-bar">
    <div class="reuse-fill" style="width:{(reuse * 100):.1f}%;background:{reuse_color};"></div>
  </div>
</div>

<h2>📋 Full JSON Report</h2>
<pre>{json.dumps(comparison, indent=2, default=str)}</pre>

</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Compare OCS-enabled vs baseline MoE workloads",
    )
    parser.add_argument(
        "--trace-dir", default="outputs/traces/ocs_comparison",
        help="Directory for trace output (default: outputs/traces/ocs_comparison)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output HTML path (default: <trace-dir>/ocs_comparison.html)",
    )
    parser.add_argument(
        "--base-config", default=os.path.join(PROJECT_DIR, "configs/compare_ocs_base.yaml"),
        help="Baseline config path",
    )
    parser.add_argument(
        "--ocs-config", default=os.path.join(PROJECT_DIR, "configs/compare_ocs_on.yaml"),
        help="OCS-enabled config path",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip running experiments; just generate report from existing traces",
    )
    args = parser.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)

    baseline_dir = os.path.join(args.trace_dir, "baseline")
    ocs_dir = os.path.join(args.trace_dir, "ocs")

    if not args.skip_run:
        print("=" * 60)
        print("Running baseline (EPS, no OCS)...")
        print("=" * 60)
        ok_base = run_experiment(args.base_config, baseline_dir)
        if not ok_base:
            print("ERROR: Baseline experiment failed. Aborting.", file=sys.stderr)
            sys.exit(1)

        print()
        print("=" * 60)
        print("Running OCS pipeline...")
        print("=" * 60)
        ok_ocs = run_experiment(args.ocs_config, ocs_dir)
        if not ok_ocs:
            print("ERROR: OCS experiment failed. Aborting.", file=sys.stderr)
            sys.exit(1)

        print()
    else:
        print("Skipping experiment runs (--skip-run), using existing traces.")

    # Load and compare
    baseline_metrics = load_trace_metrics(baseline_dir)
    ocs_metrics = load_trace_metrics(ocs_dir)

    if not baseline_metrics or not ocs_metrics:
        print("ERROR: Could not load trace metrics. Run experiments first.", file=sys.stderr)
        sys.exit(1)

    comparison = build_comparison(baseline_metrics, ocs_metrics)

    # Print JSON report
    print("=" * 60)
    print("Comparison Report (JSON)")
    print("=" * 60)
    print(json.dumps(comparison, indent=2, default=str))

    # Generate HTML report
    html = build_html_report(comparison)
    html_path = args.output or os.path.join(args.trace_dir, "ocs_comparison.html")
    with open(html_path, "w") as f:
        f.write(html)

    print(f"\nHTML report → {html_path}")

    # Quick summary
    delta = comparison.get("delta", {})
    ocs_summary = comparison.get("ocs_summary", {})
    print(f"\nSummary:")
    print(f"  Baseline total:  {baseline_metrics.get('total_us', 0):.0f} µs")
    print(f"  OCS total:       {ocs_metrics.get('total_us', 0):.0f} µs")
    if delta:
        print(f"  Delta:           {delta.get('total_us_absolute', 0):+.0f} µs ({delta.get('total_us_pct', 0):+.1f}%)")
    if ocs_summary:
        reuse = ocs_summary.get("reuse_ratio", 0)
        print(f"  Circuit reuse:   {reuse*100:.1f}%")
        print(f"  Reconfig time:   {ocs_summary.get('total_reconfig_time_us', 0):.0f} µs")


if __name__ == "__main__":
    main()
