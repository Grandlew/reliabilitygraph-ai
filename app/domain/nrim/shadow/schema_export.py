from __future__ import annotations

import json
from pathlib import Path

from .adjudication import ReviewCase
from .contracts import (
    DeploymentProfile,
    IncidentAdjudication,
    OperationalEvent,
    PredictionEnvelope,
    ServingSnapshot,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)
from .governance import FrozenPilotProtocol


CONTRACT_MODELS = {
    "telemetry_observation": TelemetryObservation,
    "operational_event": OperationalEvent,
    "topology_component": TopologyComponent,
    "topology_edge": TopologyEdge,
    "deployment_profile": DeploymentProfile,
    "serving_snapshot": ServingSnapshot,
    "prediction_envelope": PredictionEnvelope,
    "incident_adjudication": IncidentAdjudication,
    "review_case": ReviewCase,
    "frozen_pilot_protocol": FrozenPilotProtocol,
}


def export_contract_schemas(destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, model in CONTRACT_MODELS.items():
        path = destination / f"{name}.schema.json"
        path.write_text(
            json.dumps(
                model.model_json_schema(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        paths[name] = path
    return paths
