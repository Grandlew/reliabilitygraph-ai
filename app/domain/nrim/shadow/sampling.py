from __future__ import annotations

import base64
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .hashing import bytes_hash, canonical_hash, canonical_json


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class FrameSource(str, Enum):
    KNOWN_OPERATIONAL_INCIDENT = "known_operational_incident"
    NRIM_POSITIVE_EPISODE = "nrim_positive_episode"
    HEALTHY_HOUR_CANDIDATE = "healthy_hour_candidate"


class SamplingFrameRecord(StrictModel):
    case_id: str = Field(min_length=1, max_length=128)
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    hour_start_utc: datetime
    workload_band: str = Field(min_length=1, max_length=128)
    support_state: str = Field(min_length=1, max_length=128)
    source: FrameSource

    @field_validator("hour_start_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Sampling-frame time must be timezone-aware")
        return value.astimezone(timezone.utc)

    @property
    def stratum(self) -> tuple[str, str, str, str]:
        hour_band = f"{self.hour_start_utc.hour:02d}"
        return (
            self.deployment_pseudonym,
            hour_band,
            self.workload_band,
            self.support_state,
        )


class SamplingPlan(StrictModel):
    plan_id: str = Field(min_length=1, max_length=128)
    created_at_utc: datetime
    frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_commitment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[str, ...] = (
        "deployment",
        "time_of_day",
        "workload",
        "support_state",
    )
    healthy_target_per_stratum: int = Field(ge=1)
    review_all_known_incidents: bool = True
    review_all_nrim_positives: bool = True
    record_inclusion_probability: bool = True

    @field_validator("created_at_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Sampling-plan time must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def prevent_alert_only_review(self):
        required = {"deployment", "time_of_day", "workload", "support_state"}
        if set(self.strata) != required:
            raise ValueError("Sampling plan has incomplete strata")
        if not (
            self.review_all_known_incidents
            and self.review_all_nrim_positives
            and self.record_inclusion_probability
        ):
            raise ValueError("Sampling plan would permit verification bias")
        return self


class SignedSamplingPlan(StrictModel):
    plan: SamplingPlan
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature_algorithm: str = "ed25519"
    signature_base64: str


class SamplingSelection(StrictModel):
    case_id: str
    source: FrameSource
    stratum: tuple[str, str, str, str]
    inclusion_probability: float = Field(gt=0.0, le=1.0)
    selection_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SamplingResult(StrictModel):
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected: tuple[SamplingSelection, ...]
    frame_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    inclusion_probability_record_fraction: float = Field(ge=0.0, le=1.0)
    known_incident_coverage: float = Field(ge=0.0, le=1.0)
    nrim_positive_coverage: float = Field(ge=0.0, le=1.0)


class WeightedReviewOutcome(StrictModel):
    case_id: str
    inclusion_probability: float = Field(gt=0.0, le=1.0)
    exposure_hours: float = Field(gt=0.0)
    outcome: bool


def sampling_frame_hash(
    frame: tuple[SamplingFrameRecord, ...],
) -> str:
    return canonical_hash(
        [
            item.model_dump(mode="json")
            for item in sorted(frame, key=lambda row: row.case_id)
        ]
    )


def sampling_seed_commitment(seed: str) -> str:
    if len(seed) < 16:
        raise ValueError("Sampling seed must contain at least 16 characters")
    return canonical_hash({"sampling_seed": seed})


def sign_sampling_plan(
    *,
    plan: SamplingPlan,
    private_key: Ed25519PrivateKey,
    public_key_pem: bytes,
) -> SignedSamplingPlan:
    payload = plan.model_dump(mode="json")
    return SignedSamplingPlan(
        plan=plan,
        plan_sha256=canonical_hash(payload),
        public_key_sha256=bytes_hash(public_key_pem),
        signature_base64=base64.b64encode(
            private_key.sign(canonical_json(payload).encode("utf-8"))
        ).decode("ascii"),
    )


def verify_sampling_plan(
    *,
    signed: SignedSamplingPlan,
    public_key_pem: bytes,
) -> None:
    payload = signed.plan.model_dump(mode="json")
    if canonical_hash(payload) != signed.plan_sha256:
        raise ValueError("Sampling-plan commitment differs")
    if bytes_hash(public_key_pem) != signed.public_key_sha256:
        raise ValueError("Sampling-plan trust anchor differs")
    public = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(public, Ed25519PublicKey):
        raise ValueError("Sampling-plan trust anchor must be Ed25519")
    try:
        public.verify(
            base64.b64decode(signed.signature_base64, validate=True),
            canonical_json(payload).encode("utf-8"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("Sampling-plan signature is invalid") from error


def select_review_sample(
    *,
    signed_plan: SignedSamplingPlan,
    public_key_pem: bytes,
    seed: str,
    frame: tuple[SamplingFrameRecord, ...],
) -> SamplingResult:
    verify_sampling_plan(
        signed=signed_plan,
        public_key_pem=public_key_pem,
    )
    plan = signed_plan.plan
    if sampling_seed_commitment(seed) != plan.seed_commitment_sha256:
        raise ValueError("Sampling seed differs from preregistration")
    actual_frame_hash = sampling_frame_hash(frame)
    if actual_frame_hash != plan.frame_sha256:
        raise ValueError("Sampling frame differs from signed plan")
    if len({item.case_id for item in frame}) != len(frame):
        raise ValueError("Sampling-frame case IDs must be unique")

    selected: list[SamplingSelection] = []
    healthy_by_stratum: dict[
        tuple[str, str, str, str],
        list[SamplingFrameRecord],
    ] = defaultdict(list)
    known_count = 0
    positive_count = 0
    for item in frame:
        score = canonical_hash(
            {
                "seed": seed,
                "plan_id": plan.plan_id,
                "case_id": item.case_id,
            }
        )
        if item.source is FrameSource.KNOWN_OPERATIONAL_INCIDENT:
            known_count += 1
            selected.append(
                SamplingSelection(
                    case_id=item.case_id,
                    source=item.source,
                    stratum=item.stratum,
                    inclusion_probability=1.0,
                    selection_score_sha256=score,
                )
            )
        elif item.source is FrameSource.NRIM_POSITIVE_EPISODE:
            positive_count += 1
            selected.append(
                SamplingSelection(
                    case_id=item.case_id,
                    source=item.source,
                    stratum=item.stratum,
                    inclusion_probability=1.0,
                    selection_score_sha256=score,
                )
            )
        else:
            healthy_by_stratum[item.stratum].append(item)

    for stratum, rows in sorted(healthy_by_stratum.items()):
        ordered = sorted(
            rows,
            key=lambda item: canonical_hash(
                {
                    "seed": seed,
                    "plan_id": plan.plan_id,
                    "case_id": item.case_id,
                }
            ),
        )
        count = min(plan.healthy_target_per_stratum, len(ordered))
        inclusion_probability = count / len(ordered)
        for item in ordered[:count]:
            selected.append(
                SamplingSelection(
                    case_id=item.case_id,
                    source=item.source,
                    stratum=stratum,
                    inclusion_probability=inclusion_probability,
                    selection_score_sha256=canonical_hash(
                        {
                            "seed": seed,
                            "plan_id": plan.plan_id,
                            "case_id": item.case_id,
                        }
                    ),
                )
            )
    selected.sort(key=lambda item: item.case_id)
    selected_known = sum(
        item.source is FrameSource.KNOWN_OPERATIONAL_INCIDENT
        for item in selected
    )
    selected_positive = sum(
        item.source is FrameSource.NRIM_POSITIVE_EPISODE
        for item in selected
    )
    return SamplingResult(
        plan_sha256=signed_plan.plan_sha256,
        frame_sha256=actual_frame_hash,
        selected=tuple(selected),
        frame_count=len(frame),
        selected_count=len(selected),
        inclusion_probability_record_fraction=(
            sum(item.inclusion_probability > 0.0 for item in selected)
            / len(selected)
            if selected
            else 1.0
        ),
        known_incident_coverage=(
            selected_known / known_count if known_count else 1.0
        ),
        nrim_positive_coverage=(
            selected_positive / positive_count if positive_count else 1.0
        ),
    )


def inverse_probability_weighted_rate(
    outcomes: tuple[WeightedReviewOutcome, ...],
) -> dict[str, Any]:
    if not outcomes:
        return {
            "weighted_rate": None,
            "weighted_exposure_hours": 0.0,
            "effective_sample_size": 0.0,
            "assumption": (
                "No sampled outcomes; inverse-probability estimate unavailable"
            ),
        }
    weights = [
        item.exposure_hours / item.inclusion_probability
        for item in outcomes
    ]
    weighted_total = sum(weights)
    weighted_positive = sum(
        weight * float(item.outcome)
        for weight, item in zip(weights, outcomes, strict=True)
    )
    effective_sample_size = (
        sum(weights) ** 2 / sum(weight**2 for weight in weights)
    )
    return {
        "weighted_rate": weighted_positive / weighted_total,
        "weighted_exposure_hours": weighted_total,
        "effective_sample_size": effective_sample_size,
        "assumption": (
            "Horvitz-Thompson-style inverse-probability weighting assumes "
            "the signed inclusion probabilities and frame are correct."
        ),
    }
