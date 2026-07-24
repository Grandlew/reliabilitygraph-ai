from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.domain.nrim.baselines.dual_path_episode_gate import (
    DualPathConfig,
    DualPathEpisodeGate,
    fit_healthy_residual_model,
)
from app.domain.nrim.baselines.baseline_evaluator import load_split_windows
from app.domain.nrim.baselines.incident_detector import (
    choose_incident_threshold,
    fit_incident_detector,
)
from app.domain.nrim.baselines.reference_checkpoint import (
    load_reference_checkpoint,
)
from app.domain.nrim.baselines.run_dual_path_episode_experiment import (
    _probability_inputs,
    _scenario_metadata,
)
from app.domain.nrim.baselines.shift_sentinels import fit_shift_sentinels
from app.domain.nrim.baselines.temporal_episode_gate import (
    assemble_episode_records,
)
from app.domain.nrim.benchmark.dataset_loader import (
    load_model_ready_manifest,
)
from app.domain.nrim.simulation.feature_schema import (
    EDGE_TYPES,
    NODE_TYPES,
    SIGNAL_NAMES,
    build_feature_schema,
)
from app.domain.nrim.simulation.locked_test_governance import (
    verify_locked_test_seal,
)

from .bundle import (
    FrozenInferenceBundle,
    create_signed_bundle,
    generate_signing_keypair,
    serialize_models,
)
from .hashing import bytes_hash, canonical_hash, file_hash
from .schema_export import export_contract_schemas


SOURCE_FILES = (
    "app/domain/nrim/baselines/dual_path_episode_gate.py",
    "app/domain/nrim/baselines/feature_access.py",
    "app/domain/nrim/baselines/incident_detector.py",
    "app/domain/nrim/baselines/learned_fusion.py",
    "app/domain/nrim/baselines/root_cause_baselines.py",
    "app/domain/nrim/baselines/shift_sentinels.py",
    "app/domain/nrim/baselines/topology_scoring.py",
    "app/domain/nrim/simulation/feature_schema.py",
    "app/domain/nrim/simulation/graph_feature_builder.py",
    "app/domain/nrim/simulation/model_ready_exporter.py",
    "app/domain/nrim/shadow/bundle.py",
    "app/domain/nrim/shadow/contracts.py",
    "app/domain/nrim/shadow/replay.py",
    "app/domain/nrim/shadow/runtime.py",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _inference_window(window: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "window_id",
        "observation_start",
        "observation_cutoff",
        "node_ids",
        "node_feature_names",
        "node_features",
        "edge_index",
        "edge_feature_names",
        "edge_features",
    )
    return {name: window[name] for name in allowed}


def _inference_metadata(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "room_count",
        "floor_count",
        "retention_days",
        "base_occupancy_fraction",
        "catchup_recording_channels",
        "average_bitrate_mbps",
        "shared_storage",
        "redundant_middleware",
        "missing_fraction",
    )
    return {name: record.get(name) for name in fields}


def package(*, root: Path, output: Path, blueprint_path: Path) -> dict[str, Any]:
    simulation_root = (
        root / "app/domain/nrim/examples/simulation"
    )
    source_root = simulation_root / "day08_dataset_v06"
    model_root = simulation_root / "day09_model_ready_v06"
    result_path = (
        model_root
        / "benchmark_v0_6_0"
        / "dual_path_episode_experiment.json"
    )
    source_checkpoint_path = (
        simulation_root
        / "day09_model_ready_v2"
        / "benchmark_v0_5_0"
        / "reference_model_checkpoint.json"
    )
    benchmark = _load(result_path)
    seal = verify_locked_test_seal(
        seal_path=source_root / "governance/locked_test_seal.json"
    )
    development_source = _load(
        Path(seal["development_manifest_path"])
    )
    scenario_metadata = _scenario_metadata(
        development_source["records"]
    )
    manifest = load_model_ready_manifest(model_root / "manifest.json")
    development_manifest = {
        **manifest,
        "records": [
            record
            for record in manifest["records"]
            if record["split"] != "locked_test"
        ],
    }
    split_names = (
        "train",
        "validation",
        "development_test",
        "ood_test",
        "semantic_challenge",
    )
    windows_by_split = {
        split: load_split_windows(
            manifest=development_manifest,
            split=split,
        )
        for split in split_names
    }
    records, sequence_audit = assemble_episode_records(
        windows=[
            window
            for split in split_names
            for window in windows_by_split[split]
        ],
        scenario_metadata=scenario_metadata,
    )
    if not sequence_audit.leakage_free:
        raise RuntimeError("Development sequence audit is not leakage-free")
    train_records = [
        record for record in records if record.split == "train"
    ]

    incident_model = fit_incident_detector(
        training_windows=windows_by_split["train"],
        scenario_metadata=scenario_metadata,
        use_counterfactual_pairs=False,
        cohort_balancing=True,
        iterations=300,
    )
    threshold = choose_incident_threshold(
        validation_windows=windows_by_split["validation"],
        model=incident_model,
        scenario_metadata=scenario_metadata,
        maximum_false_selection_rate=0.10,
        minimum_incident_coverage=0.70,
        maximum_cohort_false_selection_rate=0.15,
    )
    expected_threshold = float(
        benchmark["stage1_window_reference"]["threshold"]
    )
    if not math.isclose(
        threshold.threshold,
        expected_threshold,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Reconstructed Stage 1 threshold differs from v0.6")

    probabilities = _probability_inputs(
        records=records,
        incident_model=incident_model,
    )
    residual_model = fit_healthy_residual_model(
        training_records=train_records,
        probabilities=probabilities,
        scenario_metadata=scenario_metadata,
    )
    if canonical_hash(asdict(residual_model)) != canonical_hash(
        benchmark["healthy_residual_model"]
    ):
        raise RuntimeError("Reconstructed healthy residual differs from v0.6")
    support_model = fit_shift_sentinels(
        training_records=train_records,
        scenario_metadata=scenario_metadata,
        registered_operational_bounds={
            "workload__log_room_count": (
                math.log1p(50.0),
                math.log1p(250.0),
            ),
            "workload__retention_days": (
                math.log1p(2.0),
                math.log1p(14.0),
            ),
            "telemetry__scenario_missing_fraction": (0.0, 0.15),
        },
    )
    if not math.isclose(
        support_model.support_threshold,
        float(benchmark["support_sentinel"]["support_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Reconstructed support sentinel differs from v0.6")
    policy = DualPathConfig(**benchmark["selected_policy"])
    if canonical_hash(asdict(policy)) != canonical_hash(
        benchmark["selected_policy"]
    ):
        raise RuntimeError("Frozen policy differs from v0.6 selection")
    _, fusion_model, _ = load_reference_checkpoint(
        path=source_checkpoint_path
    )
    if file_hash(source_checkpoint_path) != benchmark[
        "stage2_reference"
    ]["source_checkpoint_sha256"]:
        raise RuntimeError("Frozen Stage 2 checkpoint hash differs")
    schema = build_feature_schema()
    provenance = {
        "source_benchmark": "NRIM v0.6 benchmark 0.6.0",
        "source_protocol": "2.0.0",
        "source_release_status": benchmark["release_status"],
        "source_dataset_fingerprint": benchmark["dataset_fingerprint"],
        "source_benchmark_sha256": file_hash(result_path),
        "source_stage2_checkpoint_sha256": file_hash(
            source_checkpoint_path
        ),
        "source_stage2_model_sha256": benchmark[
            "stage2_reference"
        ]["learned_fusion_model_sha256"],
        "v07_blueprint_sha256": file_hash(blueprint_path),
        "git_commit_at_packaging": _git_commit(root),
        "locked_test_read_by_packager": False,
        "reconstruction_note": (
            "Stage 1/support/residual were deterministically reconstructed "
            "from development-only v0.6 train/validation artifacts because "
            "the v0.6 report did not serialize their complete weights."
        ),
    }
    policy_payload = {
        "incident_threshold": threshold.threshold,
        "dual_path_policy": asdict(policy),
        "golden_numeric_tolerance": 1e-12,
    }
    provisional = FrozenInferenceBundle(
        incident_model=incident_model,
        residual_model=residual_model,
        support_model=support_model,
        fusion_model=fusion_model,
        policy=policy,
        incident_threshold=threshold.threshold,
        feature_schema=schema,
        bundle_hash="0" * 64,
        policy_hash=canonical_hash(policy_payload),
        provenance=provenance,
        golden_snapshots=(),
    )
    validation_records = [
        record
        for record in records
        if record.split == "validation"
    ]
    selected = []
    seen_topologies = set()
    for record in validation_records:
        if record.topology_group in seen_topologies:
            continue
        selected.append(record)
        seen_topologies.add(record.topology_group)
        if len(selected) == 5:
            break
    golden = []
    for index, record in enumerate(selected):
        window = _inference_window(record.windows[0])
        metadata = _inference_metadata(
            scenario_metadata[record.scenario_id]
        )
        golden.append(
            {
                "case_id": f"golden_validation_topology_{index + 1}",
                "window": window,
                "metadata": metadata,
                "prior_logits": [],
                "expected": provisional.evaluate_window(
                    window=window,
                    metadata=metadata,
                ),
            }
        )
    artifacts = {
        "models.json": serialize_models(
            incident_model=incident_model,
            residual_model=residual_model,
            support_model=support_model,
            fusion_model=fusion_model,
        ),
        "policy.json": policy_payload,
        "feature_schema.json": schema.model_dump(mode="json"),
        "categories.json": {
            "node_types": NODE_TYPES,
            "edge_types": EDGE_TYPES,
            "signal_names": SIGNAL_NAMES,
        },
        "golden_snapshots.json": golden,
        "provenance.json": provenance,
    }
    output.mkdir(parents=True, exist_ok=False)
    private_key, public_pem = generate_signing_keypair()
    bundle_path = output / "frozen_v06.nrimbundle"
    public_path = output / "frozen_v06_public_key.pem"
    bundle_hash = create_signed_bundle(
        output_path=bundle_path,
        public_key_path=public_path,
        private_key=private_key,
        public_key_pem=public_pem,
        artifacts=artifacts,
        source_root=root,
        source_paths=tuple(root / item for item in SOURCE_FILES),
        provenance=provenance,
    )
    schemas = export_contract_schemas(output / "schemas")
    commitment = {
        "bundle_sha256": file_hash(bundle_path),
        "bundle_manifest_sha256": bundle_hash,
        "public_key_sha256": bytes_hash(public_pem),
        "feature_schema_sha256": canonical_hash(
            schema.model_dump(mode="json")
        ),
        "policy_sha256": canonical_hash(policy_payload),
        "source_benchmark_sha256": file_hash(result_path),
        "schema_files": {
            name: file_hash(path)
            for name, path in sorted(schemas.items())
        },
        "truth_status": (
            "frozen_candidate_bundle_for_prospective_shadow_evaluation"
        ),
    }
    (output / "bundle_commitment.json").write_text(
        json.dumps(commitment, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return commitment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "app/domain/nrim/examples/shadow/v0_7_0/frozen_bundle"
        ),
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=Path(
            r"C:\Users\Grand Master\Downloads"
            r"\NRIM_v0.7_Prospective_Shadow_Evidence_Implementation_Blueprint.docx"
        ),
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    commitment = package(
        root=root,
        output=(root / arguments.output).resolve(),
        blueprint_path=arguments.blueprint.resolve(),
    )
    print(json.dumps(commitment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
