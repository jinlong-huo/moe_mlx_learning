"""Per-rank logging to avoid interleaved output."""
import sys


def log(rank: int, msg: str) -> None:
    print(f"[rank={rank:02d}] {msg}", file=sys.stderr, flush=True)


def log_summary(rank: int, metrics: dict) -> None:
    parts = "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    log(rank, f"SUMMARY  {parts}")
