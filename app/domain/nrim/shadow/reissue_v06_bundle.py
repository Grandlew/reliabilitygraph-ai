from __future__ import annotations

import argparse
import base64
import json
import zipfile
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)

from .bundle import (
    REQUIRED_ARTIFACTS,
    create_signed_bundle,
    generate_signing_keypair,
)
from .hashing import bytes_hash, canonical_hash, canonical_json, file_hash
from .package_v06_bundle import SOURCE_FILES
from .schema_export import export_contract_schemas


EXPECTED_CONTRACT_CORRECTIONS = {
    "app/domain/nrim/shadow/contracts.py",
    "app/domain/nrim/shadow/replay.py",
}


def _verified_prior_artifacts(
    *,
    root: Path,
    prior: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    commitment = json.loads(
        (prior / "bundle_commitment.json").read_text(encoding="utf-8")
    )
    bundle_path = prior / "frozen_v06.nrimbundle"
    public_key_path = prior / "frozen_v06_public_key.pem"
    if file_hash(bundle_path) != commitment["bundle_sha256"]:
        raise RuntimeError("Prior bundle file differs from its commitment")
    public_key_bytes = public_key_path.read_bytes()
    if bytes_hash(public_key_bytes) != commitment["public_key_sha256"]:
        raise RuntimeError("Prior public key differs from its commitment")
    public_key = serialization.load_pem_public_key(public_key_bytes)
    if not isinstance(public_key, Ed25519PublicKey):
        raise RuntimeError("Prior trust anchor is not Ed25519")

    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Prior bundle contains duplicate entries")
        if set(names) != {"manifest.json", *REQUIRED_ARTIFACTS}:
            raise RuntimeError("Prior bundle has an unexpected entry set")
        manifest = json.loads(archive.read("manifest.json"))
        signed = manifest["signed_payload"]
        if canonical_hash(signed) != commitment["bundle_manifest_sha256"]:
            raise RuntimeError("Prior signed manifest differs from commitment")
        if signed["public_key_sha256"] != bytes_hash(public_key_bytes):
            raise RuntimeError("Prior manifest names a different trust anchor")
        try:
            public_key.verify(
                base64.b64decode(
                    manifest["signature_base64"],
                    validate=True,
                ),
                canonical_json(signed).encode("utf-8"),
            )
        except (InvalidSignature, ValueError) as error:
            raise RuntimeError("Prior bundle signature is invalid") from error
        artifacts: dict[str, Any] = {}
        for name in REQUIRED_ARTIFACTS:
            raw = archive.read(name)
            if bytes_hash(raw) != signed["artifact_hashes"][name]:
                raise RuntimeError(
                    f"Prior bundle artifact differs: {name}"
                )
            artifacts[name] = json.loads(raw)

    changed_sources = {
        relative
        for relative, expected in signed["source_hashes"].items()
        if file_hash(root / relative) != expected
    }
    if changed_sources != EXPECTED_CONTRACT_CORRECTIONS:
        raise RuntimeError(
            "Prior bundle source delta is not the registered quality-contract "
            f"correction: {sorted(changed_sources)!r}"
        )
    return artifacts, signed, commitment


def reissue(*, root: Path, prior: Path, output: Path) -> dict[str, Any]:
    artifacts, prior_signed, prior_commitment = (
        _verified_prior_artifacts(root=root, prior=prior)
    )
    prior_provenance = artifacts["provenance.json"]
    provenance = {
        **prior_provenance,
        "bundle_reissue": {
            "prior_bundle_sha256": prior_commitment["bundle_sha256"],
            "prior_bundle_manifest_sha256": (
                prior_commitment["bundle_manifest_sha256"]
            ),
            "reason": (
                "The prospective telemetry contract now carries the frozen "
                "v0.6 observation-quality category required by "
                "low_quality_fraction features."
            ),
            "model_artifacts_reused_without_refit": True,
            "policy_artifact_reused_without_change": True,
            "locked_test_read_by_reissue": False,
            "changed_source_paths": sorted(
                EXPECTED_CONTRACT_CORRECTIONS
            ),
        },
    }
    artifacts["provenance.json"] = provenance
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
        "feature_schema_sha256": prior_commitment[
            "feature_schema_sha256"
        ],
        "policy_sha256": prior_commitment["policy_sha256"],
        "source_benchmark_sha256": prior_commitment[
            "source_benchmark_sha256"
        ],
        "prior_bundle_manifest_sha256": canonical_hash(prior_signed),
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
        "--prior",
        type=Path,
        default=Path(
            "app/domain/nrim/examples/shadow/v0_7_0/frozen_bundle"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "app/domain/nrim/examples/shadow/v0_7_0/"
            "frozen_bundle_candidate"
        ),
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    commitment = reissue(
        root=root,
        prior=(root / arguments.prior).resolve(),
        output=(root / arguments.output).resolve(),
    )
    print(json.dumps(commitment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
