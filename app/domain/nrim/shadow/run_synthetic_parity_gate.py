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

from .parity import verify_synthetic_serving_parity


def main() -> None:
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
    first_by_scenario = {}
    for window in sorted(
        windows,
        key=lambda item: (
            item["source_scenario_id"],
            item["observation_cutoff"],
        ),
    ):
        first_by_scenario.setdefault(
            str(window["source_scenario_id"]),
            window,
        )
    results = []
    seen_topologies = set()
    for scenario_id, window in first_by_scenario.items():
        row = metadata[scenario_id]
        topology = str(row["topology_fingerprint"])
        if topology in seen_topologies:
            continue
        seen_topologies.add(topology)
        observable = json.loads(
            Path(row["observable_path"]).read_text(encoding="utf-8")
        )
        results.append(
            verify_synthetic_serving_parity(
                observable_scenario=observable,
                scenario_metadata=row,
                reference_window=window,
                pseudonymization_secret=b"synthetic-parity-key-" + b"x" * 32,
            )
        )
        if len(results) == 5:
            break
    report = {
        "gate": "synthetic_serving_feature_parity",
        "locked_test_read": False,
        "case_count": len(results),
        "matching_case_count": sum(
            item["matches"] for item in results
        ),
        "parity_fraction": (
            sum(item["matches"] for item in results) / len(results)
            if results
            else 0.0
        ),
        "cases": results,
        "truth_note": (
            "Development-only synthetic parity is necessary but does not "
            "establish real telemetry availability or semantic parity."
        ),
    }
    output = (
        root
        / "app/domain/nrim/examples/shadow/v0_7_0"
        / "synthetic_serving_parity.json"
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["parity_fraction"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
