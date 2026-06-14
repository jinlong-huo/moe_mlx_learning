"""Entry point: parse args, load config, launch the experiment.

Two usage modes:
  1. Direct:  python -m src.launcher --config configs/synthetic_moe.yaml
  2. Script:  bash scripts/run_synthetic.sh
"""
from src.launcher import load_config, launch
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Communication Research")
    parser.add_argument("--config", type=str, default="configs/synthetic_moe.yaml")
    parser.add_argument("--trace-dir", type=str, default="outputs/traces")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[main] loaded config: {args.config}")
    print(f"  world_size={config.get('world_size')}, mode={config.get('runtime', {}).get('mode')}")

    launch(config, trace_dir=args.trace_dir)
