"""OCS-aware topology configuration and pool management.

OcsTopology wraps an OcsCircuitPool with configuration. It is separate from
the existing Topology (src/comm/topology.py) — OCS is an overlay that can
coexist with hierarchical topology modeling.

When both OCS and hierarchical topology are enabled, the Transport's
_inject_delay uses the OCS model (which may be faster or slower depending
on circuit cache state). The hierarchical topology can be used as a fallback
or combined with OCS in future iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ocs.circuit import OcsCircuitPool, OcsPoolMetrics


@dataclass
class OcsTopologyConfig:
    """Configuration for OCS circuit topology.

    Attributes:
        enabled: master on/off switch — when False, no OCS code runs
        max_circuits: maximum simultaneous optical circuits per rank
        reconfig_time_us: circuit establishment time (MEMS: 1-50us,
                          beam-steering: 10-1000us)
        circuit_latency_us: base optical path latency (usually 1-5us)
        circuit_bandwidth_gbps: per-circuit bandwidth once established
        placement_strategy: "round_robin" (default) or "affinity"
                           (expert co-activation aware)
    """
    enabled: bool = False
    max_circuits: int = 32
    reconfig_time_us: float = 50.0
    circuit_latency_us: float = 1.0
    circuit_bandwidth_gbps: float = 200.0
    placement_strategy: str = "round_robin"


class OcsTopology:
    """Holds OCS configuration and circuit pool.

    Created once per rank in worker.py. When enabled, the pool is passed
    to Transport for delay injection and to the scheduler for circuit
    pre-establishment.
    """

    def __init__(self, config: OcsTopologyConfig):
        self.config = config
        self.pool: Optional[OcsCircuitPool] = None
        if config.enabled:
            self.pool = OcsCircuitPool(
                max_circuits=config.max_circuits,
                reconfig_time_us=config.reconfig_time_us,
                circuit_latency_us=config.circuit_latency_us,
                circuit_bw_gbps=config.circuit_bandwidth_gbps,
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.pool is not None

    def get_pool_metrics(self) -> Optional[OcsPoolMetrics]:
        """Return accumulated circuit pool metrics, or None if disabled."""
        if self.pool is None:
            return None
        return self.pool.metrics
