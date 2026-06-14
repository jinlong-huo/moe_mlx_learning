#!/usr/bin/env bash
# run_synthetic.sh — Stage 1: Run the synthetic MoE baseline (serial mode)
#
# This script runs the minimum viable experiment:
#   2 processes, 2 experts, fixed routing, synthetic tensors.
#   Serial mode first (baseline), then override to overlap mode for comparison.
#
# Prerequisites:
#   pip install torch pyyaml
#
# Usage:
#   bash scripts/run_synthetic.sh          # serial baseline
#   bash scripts/run_synthetic.sh overlap  # overlap mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

MODE="${1:-serial}"

echo "=== MoE Research: Synthetic Benchmark ==="
echo "  Mode:       ${MODE}"
echo "  Config:     configs/synthetic_moe.yaml"
echo "  Project:    ${PROJECT_DIR}"
echo ""

# Override runtime mode via env var (the launcher reads config, we use sed)
# Simpler approach: create a temp config with the desired mode
CONFIG="configs/synthetic_moe.yaml"

if [ "$MODE" = "overlap" ]; then
    echo "[run] Switching to overlap mode..."
    # Use Python to modify the config on-the-fly
    python3 -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['runtime']['mode'] = 'overlap'
with open('${CONFIG}.tmp', 'w') as f:
    yaml.dump(cfg, f)
"
    CONFIG="${CONFIG}.tmp"
fi

echo "[run] Launching workers..."
python3 -m src.launcher --config "$CONFIG" --trace-dir outputs/traces

# Cleanup temp config
rm -f "configs/synthetic_moe.yaml.tmp"

# Merge per-rank traces for chrome://tracing / Perfetto
echo "[run] Merging traces..."
python3 scripts/merge_traces.py outputs/traces/ -o outputs/traces/merged_trace.json

# Generate standalone interactive HTML viewer
echo "[run] Generating trace viewer..."
python3 scripts/trace_viz.py outputs/traces/ -o outputs/traces/trace_viewer.html

echo ""
echo "[run] Done. Traces written to outputs/traces/"
echo "[run] Interactive:  open outputs/traces/trace_viewer.html"
echo "[run] Chrome trace: open chrome://tracing and load outputs/traces/merged_trace.json"
