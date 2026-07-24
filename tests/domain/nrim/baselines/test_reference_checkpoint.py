from __future__ import annotations

import json

import pytest

from app.domain.nrim.baselines.incident_detector import (
    IncidentDetectorModel,
)
from app.domain.nrim.baselines.learned_fusion import (
    RootCauseFusionModel,
)
from app.domain.nrim.baselines.reference_checkpoint import (
    build_reference_checkpoint,
    load_reference_checkpoint,
    save_reference_checkpoint,
)


def test_reference_checkpoint_round_trip_and_hash(tmp_path) -> None:
    incident = IncidentDetectorModel(
        feature_names=("signal",),
        means=(1.0,),
        scales=(2.0,),
        weights=(0.5,),
        bias=-0.25,
        ood_distance_threshold=3.0,
    )
    fusion = RootCauseFusionModel(
        feature_names=("anomaly",),
        means=(0.0,),
        scales=(1.0,),
        weights=(2.0,),
        bias=0.1,
        use_topology=True,
    )
    path = tmp_path / "checkpoint.json"
    checkpoint = build_reference_checkpoint(
        incident_model=incident,
        fusion_model=fusion,
        incident_threshold=0.6,
        dataset_fingerprint="a" * 64,
    )
    save_reference_checkpoint(checkpoint=checkpoint, path=path)

    loaded_incident, loaded_fusion, threshold = (
        load_reference_checkpoint(
            path=path,
            expected_dataset_fingerprint="a" * 64,
        )
    )

    assert loaded_incident == incident
    assert loaded_fusion == fusion
    assert threshold == 0.6

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["incident_threshold"] = 0.7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_reference_checkpoint(path=path)

