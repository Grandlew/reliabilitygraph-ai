"""NRIM v0.7 prospective shadow evidence system.

The package is deliberately read-only with respect to IPTV infrastructure.
It validates mirrored observations, reconstructs event-time snapshots, runs
the frozen v0.6 inference policy, and records immutable evidence for later
independent adjudication.
"""

from .contracts import (
    Applicability,
    DecisionState,
    IncidentAdjudication,
    ObservationQuality,
    PredictionEnvelope,
    ServingSnapshot,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)

__all__ = [
    "Applicability",
    "DecisionState",
    "IncidentAdjudication",
    "ObservationQuality",
    "PredictionEnvelope",
    "ServingSnapshot",
    "TelemetryObservation",
    "TopologyComponent",
    "TopologyEdge",
]
