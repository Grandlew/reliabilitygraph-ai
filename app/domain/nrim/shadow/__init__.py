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
from .decision_events import DecisionEvent, DecisionEventType
from .decision_projection import (
    DecisionComparison,
    DecisionSnapshot,
    project_decision,
)

__all__ = [
    "Applicability",
    "DecisionState",
    "DecisionComparison",
    "DecisionEvent",
    "DecisionEventType",
    "DecisionSnapshot",
    "IncidentAdjudication",
    "ObservationQuality",
    "PredictionEnvelope",
    "ServingSnapshot",
    "TelemetryObservation",
    "TopologyComponent",
    "TopologyEdge",
    "project_decision",
]
