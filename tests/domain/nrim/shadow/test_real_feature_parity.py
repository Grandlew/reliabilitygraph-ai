from __future__ import annotations

import json
from pathlib import Path

from app.domain.nrim.baselines.baseline_evaluator import load_split_windows
from app.domain.nrim.benchmark.dataset_loader import (
    load_model_ready_manifest,
)
from app.domain.nrim.simulation.locked_test_governance import (
    verify_locked_test_seal,
)
from app.domain.nrim.shadow.parity import (
    verify_synthetic_serving_parity,
)


def test_strict_shadow_contract_reproduces_exported_v06_tensor() -> None:
    root = Path(__file__).resolve().parents[4]
    simulation = root / "app/domain/nrim/examples/simulation"
    source = simulation / "day08_dataset_v06"
    model = simulation / "day09_model_ready_v06"
    seal = verify_locked_test_seal(
        seal_path=source / "governance/locked_test_seal.json"
    )
    development = json.loads(
        Path(seal["development_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    metadata = {
        str(item["scenario_id"]): item
        for item in development["records"]
    }
    manifest = load_model_ready_manifest(model / "manifest.json")
    development_manifest = {
        **manifest,
        "records": [
            item
            for item in manifest["records"]
            if item["split"] != "locked_test"
        ],
    }
    windows = load_split_windows(
        manifest=development_manifest,
        split="validation",
    )
    reference = min(
        windows,
        key=lambda item: (
            item["source_scenario_id"],
            item["observation_cutoff"],
        ),
    )
    row = metadata[str(reference["source_scenario_id"])]
    observable = json.loads(
        Path(row["observable_path"]).read_text(encoding="utf-8")
    )
    result = verify_synthetic_serving_parity(
        observable_scenario=observable,
        scenario_metadata=row,
        reference_window=reference,
        pseudonymization_secret=b"test-parity-" + b"x" * 32,
    )
    assert result["matches"], json.dumps(
        result["first_differences"],
        indent=2,
    )
