from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Self

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

from .evaluation import exact_poisson_rate_upper
from .hashing import bytes_hash, canonical_hash, canonical_json


class EvidencePhase(str, Enum):
    REAL_REPLAY = "real_replay"
    SILENT_SHADOW = "silent_shadow"
    OPERATOR_ADVISORY = "operator_advisory"
    OPERATIONAL_CANDIDATE = "operational_candidate"


class EstimandName(str, Enum):
    INCIDENT_RECALL = "incident_recall"
    SEVERE_RECALL = "severe_recall"
    ALERT_BURDEN = "alert_burden"
    ROOT_CAUSE_RANKING = "root_cause_ranking"
    SUPPORT_SAFETY = "support_safety"
    PATH_ATTRIBUTION = "path_attribution"
    HUMAN_UTILITY = "human_utility"


class InferenceIntent(str, Enum):
    DESCRIPTIVE = "descriptive"
    CONFIRMATORY = "confirmatory"


class ThresholdDirection(str, Enum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class UncertaintyMethod(str, Enum):
    EXACT_BINOMIAL_ONE_SIDED = "exact_binomial_one_sided"
    EXACT_POISSON_ONE_SIDED = "exact_poisson_one_sided"
    DEPLOYMENT_CLUSTER_SENSITIVITY = (
        "deployment_cluster_sensitivity"
    )
    DEPLOYMENT_STRATIFIED_BOOTSTRAP = (
        "deployment_stratified_bootstrap"
    )
    PAIRED_ROLLOUT_AWARE = "paired_rollout_aware"
    EXACT_DEPLOYMENT_INTERVALS = "exact_deployment_intervals"
    RANDOM_EFFECTS_SENSITIVITY = "random_effects_sensitivity"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class EstimandSpec(StrictModel):
    name: EstimandName
    evidence_phase: EvidencePhase
    target_population: str = Field(min_length=10, max_length=1000)
    primary_unit: str = Field(min_length=3, max_length=256)
    denominator: str = Field(min_length=10, max_length=1000)
    inclusion_rule: str = Field(min_length=10, max_length=2000)
    adjudication_status: str = Field(min_length=3, max_length=512)
    point_metric: str = Field(min_length=3, max_length=256)
    point_direction: ThresholdDirection
    point_threshold: float
    intent: InferenceIntent
    uncertainty_methods: tuple[UncertaintyMethod, ...]
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    bound_direction: ThresholdDirection | None = None
    bound_threshold: float | None = None
    minimum_count: int = Field(default=0, ge=0)
    subgroup_minimum_count: int = Field(default=0, ge=0)
    coverage_denominator_reported: bool = True
    assumptions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inference_contract(self) -> Self:
        if len(set(self.uncertainty_methods)) != len(
            self.uncertainty_methods
        ):
            raise ValueError("Uncertainty methods must be unique")
        if self.intent is InferenceIntent.CONFIRMATORY:
            if self.bound_direction is None or self.bound_threshold is None:
                raise ValueError(
                    "Confirmatory estimands require a bound gate"
                )
            if not self.uncertainty_methods:
                raise ValueError(
                    "Confirmatory estimands require uncertainty"
                )
        elif (
            self.bound_direction is not None
            or self.bound_threshold is not None
        ):
            raise ValueError(
                "Descriptive estimands cannot silently define a bound gate"
            )
        if (
            self.name is EstimandName.ROOT_CAUSE_RANKING
            and not self.coverage_denominator_reported
        ):
            raise ValueError("Ranking must report its coverage denominator")
        return self


class NegativeSamplingDesign(StrictModel):
    design_id: str = Field(min_length=1, max_length=128)
    seed_commitment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[str, ...]
    minimum_inclusion_probability: float = Field(gt=0.0, le=1.0)
    review_all_known_incidents: bool = True
    review_all_nrim_positive_episodes: bool = True
    preserve_unresolved_labels: bool = True
    record_inclusion_probabilities: bool = True
    double_review_rule: str = Field(min_length=10, max_length=1000)

    @model_validator(mode="after")
    def prevent_verification_bias(self) -> Self:
        required = {"deployment", "time_of_day", "workload", "support_state"}
        if not required.issubset(self.strata):
            raise ValueError(
                "Negative sampling must stratify deployment, time, "
                "workload and support state"
            )
        if not (
            self.review_all_known_incidents
            and self.review_all_nrim_positive_episodes
            and self.preserve_unresolved_labels
            and self.record_inclusion_probabilities
        ):
            raise ValueError(
                "Sampling design would create verification or denominator bias"
            )
        return self


class StopRule(StrictModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")
    description: str = Field(min_length=10, max_length=1000)
    accountable_role: str = Field(min_length=3, max_length=128)
    automatic_pause: bool
    preserve_evidence: bool = True
    recovery_acceptance: str = Field(min_length=10, max_length=1000)


class SeparationOfDuties(StrictModel):
    protocol_owner: str = Field(min_length=12, max_length=128)
    evidence_custodian: str = Field(min_length=12, max_length=128)
    review_lead: str = Field(min_length=12, max_length=128)
    release_authority: str = Field(min_length=12, max_length=128)
    operations_owner: str = Field(min_length=12, max_length=128)
    privacy_owner: str = Field(min_length=12, max_length=128)
    security_owner: str = Field(min_length=12, max_length=128)

    @model_validator(mode="after")
    def require_distinct_people(self) -> Self:
        assignments = tuple(self.model_dump().values())
        if len(set(assignments)) != len(assignments):
            raise ValueError(
                "Protocol, evidence, review, release, operations, privacy "
                "and security duties must be separately assigned"
            )
        return self


class StatisticalContract(StrictModel):
    estimands: tuple[EstimandSpec, ...]
    sampling_design: NegativeSamplingDesign
    minimum_independent_deployments: int = Field(default=5, ge=5)
    minimum_topology_families: int = Field(default=5, ge=5)
    minimum_healthy_deployment_hours: float = Field(
        default=3000.0,
        ge=3000.0,
    )
    minimum_adjudicated_incidents: int = Field(default=30, ge=30)
    minimum_silent_weeks: float = Field(default=8.0, ge=8.0)
    minimum_severe_incidents: int = Field(default=14, ge=1)
    minimum_deployments_for_cluster_claim: int = Field(default=10, ge=10)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)

    @model_validator(mode="after")
    def validate_estimands(self) -> Self:
        names = [item.name for item in self.estimands]
        if len(names) != len(set(names)):
            raise ValueError("Estimands must be unique")
        required = {
            EstimandName.INCIDENT_RECALL,
            EstimandName.SEVERE_RECALL,
            EstimandName.ALERT_BURDEN,
            EstimandName.ROOT_CAUSE_RANKING,
            EstimandName.SUPPORT_SAFETY,
            EstimandName.PATH_ATTRIBUTION,
        }
        if not required.issubset(names):
            raise ValueError(
                "Statistical contract omits a core silent-phase estimand"
            )
        severe = next(
            item
            for item in self.estimands
            if item.name is EstimandName.SEVERE_RECALL
        )
        if (
            severe.intent is InferenceIntent.CONFIRMATORY
            and severe.bound_threshold is not None
        ):
            best_possible = self.alpha ** (
                1.0 / self.minimum_severe_incidents
            )
            if best_possible < severe.bound_threshold:
                raise ValueError(
                    "Severe confirmatory bound is impossible at the "
                    "registered minimum count"
                )
        return self


class OELRProtocol(StrictModel):
    schema_version: str = "0.7.1-oelr"
    protocol_id: str = Field(min_length=1, max_length=128)
    registered_at_utc: datetime
    target_phase: EvidencePhase
    planned_start_utc: datetime
    planned_end_utc: datetime
    deployment_pseudonyms: tuple[str, ...] = Field(min_length=5)
    topology_family_count: int = Field(ge=5)
    model_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_public_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    watermark_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    historical_replay_plan_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    dress_rehearsal_plan_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    review_workflow_plan_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    external_anchor_expectation_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    collector_contract_hashes: dict[str, str] = Field(min_length=5)
    topology_contract_hashes: dict[str, str] = Field(min_length=5)
    statistical_contract: StatisticalContract
    stop_rules: tuple[StopRule, ...] = Field(min_length=8)
    path_retention_rules: tuple[str, ...] = Field(min_length=3)
    duties: SeparationOfDuties
    outcomes_inspected_before_registration: bool = False
    model_or_policy_tuning_allowed: bool = False
    predictions_hidden_from_frontline: bool = True
    automatic_remediation_enabled: bool = False

    @field_validator(
        "registered_at_utc",
        "planned_start_utc",
        "planned_end_utc",
    )
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Protocol timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator(
        "collector_contract_hashes",
        "topology_contract_hashes",
    )
    @classmethod
    def validate_hash_map(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value.values()
        ):
            raise ValueError("Contract commitments must be SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if self.registered_at_utc > self.planned_start_utc:
            raise ValueError("Protocol must be registered before data access")
        if self.planned_start_utc >= self.planned_end_utc:
            raise ValueError("Protocol end must follow its start")
        if self.outcomes_inspected_before_registration:
            raise ValueError("Outcomes were inspected before preregistration")
        if self.model_or_policy_tuning_allowed:
            raise ValueError("v0.7 OELR must prohibit outcome-driven tuning")
        if self.automatic_remediation_enabled:
            raise ValueError("OELR cannot enable automatic remediation")
        if (
            self.target_phase is EvidencePhase.SILENT_SHADOW
            and not self.predictions_hidden_from_frontline
        ):
            raise ValueError("Silent predictions must remain hidden")
        names = {
            item.name: item
            for item in self.statistical_contract.estimands
        }
        human = names.get(EstimandName.HUMAN_UTILITY)
        if self.target_phase is EvidencePhase.SILENT_SHADOW and human:
            raise ValueError(
                "Human utility is unavailable while predictions are hidden"
            )
        if (
            self.target_phase
            in {
                EvidencePhase.OPERATOR_ADVISORY,
                EvidencePhase.OPERATIONAL_CANDIDATE,
            }
            and human is None
        ):
            raise ValueError("Advisory protocols require human utility")
        if len(set(self.deployment_pseudonyms)) != len(
            self.deployment_pseudonyms
        ):
            raise ValueError("Deployment pseudonyms must be unique")
        if set(self.collector_contract_hashes) != set(
            self.deployment_pseudonyms
        ):
            raise ValueError(
                "Every deployment needs one collector-contract commitment"
            )
        if set(self.topology_contract_hashes) != set(
            self.deployment_pseudonyms
        ):
            raise ValueError(
                "Every deployment needs one topology-contract commitment"
            )
        if len({item.code for item in self.stop_rules}) != len(
            self.stop_rules
        ):
            raise ValueError("Stop-rule codes must be unique")
        return self


class CompiledProtocol(StrictModel):
    protocol: OELRProtocol
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_version: str = "0.7.1"
    derived_constraints: dict[str, int | float | bool | str]


def _maximum_false_episodes(
    *,
    exposure: float,
    upper_rate_per_100_hours: float,
    confidence: float,
) -> int:
    events = 0
    while (
        100.0
        * exact_poisson_rate_upper(
            events=events,
            exposure=exposure,
            confidence=confidence,
        )
        <= upper_rate_per_100_hours
    ):
        events += 1
    return events - 1


def compile_protocol(protocol: OELRProtocol) -> CompiledProtocol:
    contract = protocol.statistical_contract
    burden = next(
        item
        for item in contract.estimands
        if item.name is EstimandName.ALERT_BURDEN
    )
    recall = next(
        item
        for item in contract.estimands
        if item.name is EstimandName.INCIDENT_RECALL
    )
    maximum_false = -1
    if (
        burden.intent is InferenceIntent.CONFIRMATORY
        and burden.bound_threshold is not None
    ):
        maximum_false = _maximum_false_episodes(
            exposure=contract.minimum_healthy_deployment_hours,
            upper_rate_per_100_hours=burden.bound_threshold,
            confidence=burden.confidence,
        )
    derived = {
        "human_utility_required": (
            protocol.target_phase
            in {
                EvidencePhase.OPERATOR_ADVISORY,
                EvidencePhase.OPERATIONAL_CANDIDATE,
            }
        ),
        "predictions_must_be_hidden": (
            protocol.target_phase is EvidencePhase.SILENT_SHADOW
        ),
        "maximum_false_episodes_at_minimum_exposure": maximum_false,
        "incident_recall_requires_bound": (
            recall.intent is InferenceIntent.CONFIRMATORY
        ),
        "cluster_inference_status": (
            "feasibility_only_below_10_deployments"
        ),
        "model_tuning_permitted": False,
        "automatic_remediation_permitted": False,
    }
    return CompiledProtocol(
        protocol=protocol,
        protocol_sha256=canonical_hash(
            protocol.model_dump(mode="json")
        ),
        derived_constraints=derived,
    )


def default_statistical_contract(
    *,
    target_phase: EvidencePhase,
    sampling_seed_commitment_sha256: str,
) -> StatisticalContract:
    estimands = [
        EstimandSpec(
            name=EstimandName.INCIDENT_RECALL,
            evidence_phase=EvidencePhase.SILENT_SHADOW,
            target_population=(
                "All preregistered real IPTV incidents in included "
                "deployments and dates"
            ),
            primary_unit="adjudicated incident episode",
            denominator=(
                "All included incidents, including incidents for which NRIM "
                "produced no alert"
            ),
            inclusion_rule=(
                "Every known operational incident is reviewed independently "
                "of NRIM output"
            ),
            adjudication_status="latest resolved or explicitly unresolved label",
            point_metric="episode_recall",
            point_direction=ThresholdDirection.AT_LEAST,
            point_threshold=0.80,
            intent=InferenceIntent.CONFIRMATORY,
            uncertainty_methods=(
                UncertaintyMethod.EXACT_BINOMIAL_ONE_SIDED,
                UncertaintyMethod.DEPLOYMENT_CLUSTER_SENSITIVITY,
            ),
            bound_direction=ThresholdDirection.AT_LEAST,
            bound_threshold=0.80,
            minimum_count=30,
            subgroup_minimum_count=5,
            assumptions=(
                "Incidents are inclusion-complete within registered sources.",
                "The exact bound conditions on the observed incident set.",
                "Cluster sensitivity resamples whole deployments.",
            ),
        ),
        EstimandSpec(
            name=EstimandName.SEVERE_RECALL,
            evidence_phase=EvidencePhase.SILENT_SHADOW,
            target_population=(
                "All preregistered high or critical real IPTV incidents"
            ),
            primary_unit="severe adjudicated incident episode",
            denominator="All included severe incidents whether alerted or not",
            inclusion_rule="Severity is assigned blinded to NRIM output",
            adjudication_status="confirmed or probable severity label",
            point_metric="severe_recall",
            point_direction=ThresholdDirection.AT_LEAST,
            point_threshold=1.0,
            intent=InferenceIntent.CONFIRMATORY,
            uncertainty_methods=(
                UncertaintyMethod.EXACT_BINOMIAL_ONE_SIDED,
            ),
            bound_direction=ThresholdDirection.AT_LEAST,
            bound_threshold=0.80,
            minimum_count=14,
            assumptions=(
                "At least fourteen severe incidents are required.",
                "No detected severe incident is excluded post hoc.",
            ),
        ),
        EstimandSpec(
            name=EstimandName.ALERT_BURDEN,
            evidence_phase=EvidencePhase.SILENT_SHADOW,
            target_population=(
                "Healthy deployment time in registered deployments and dates"
            ),
            primary_unit="healthy deployment-hour",
            denominator=(
                "All eligible healthy deployment-hours with sampled-negative "
                "inclusion probabilities retained"
            ),
            inclusion_rule=(
                "Every NRIM-positive episode is actionability-reviewed and "
                "healthy hours follow the signed stratified design"
            ),
            adjudication_status="actionability adjudicated or unresolved",
            point_metric="episodes_per_100_healthy_hours",
            point_direction=ThresholdDirection.AT_MOST,
            point_threshold=0.10,
            intent=InferenceIntent.CONFIRMATORY,
            uncertainty_methods=(
                UncertaintyMethod.EXACT_POISSON_ONE_SIDED,
                UncertaintyMethod.EXACT_DEPLOYMENT_INTERVALS,
                UncertaintyMethod.DEPLOYMENT_CLUSTER_SENSITIVITY,
                UncertaintyMethod.RANDOM_EFFECTS_SENSITIVITY,
            ),
            bound_direction=ThresholdDirection.AT_MOST,
            bound_threshold=0.20,
            assumptions=(
                "The exact count bound assumes a Poisson event count.",
                "Deployment heterogeneity is reported rather than hidden.",
                "Random effects are sensitivity only when clusters suffice.",
            ),
        ),
        EstimandSpec(
            name=EstimandName.ROOT_CAUSE_RANKING,
            evidence_phase=EvidencePhase.SILENT_SHADOW,
            target_population=(
                "In-support incidents with an adjudicable component root cause"
            ),
            primary_unit="rankable incident episode",
            denominator=(
                "All in-support incidents with a root-cause label; ranking "
                "coverage is reported separately"
            ),
            inclusion_rule=(
                "Root-cause availability is determined independently of rank"
            ),
            adjudication_status="confirmed component or unresolved",
            point_metric="mrr_and_hits_at_k",
            point_direction=ThresholdDirection.AT_LEAST,
            point_threshold=0.70,
            intent=InferenceIntent.DESCRIPTIVE,
            uncertainty_methods=(
                UncertaintyMethod.DEPLOYMENT_STRATIFIED_BOOTSTRAP,
            ),
            assumptions=(
                "Candidate-set coverage is an explicit denominator.",
            ),
        ),
        EstimandSpec(
            name=EstimandName.SUPPORT_SAFETY,
            evidence_phase=EvidencePhase.SILENT_SHADOW,
            target_population="Every real decision snapshot",
            primary_unit="decision snapshot",
            denominator=(
                "All snapshots including unsupported and quarantined inputs"
            ),
            inclusion_rule="No support or data-quality state is excluded",
            adjudication_status="not label-dependent",
            point_metric="safe_semantics_rate",
            point_direction=ThresholdDirection.AT_LEAST,
            point_threshold=1.0,
            intent=InferenceIntent.DESCRIPTIVE,
            uncertainty_methods=(
                UncertaintyMethod.EXACT_DEPLOYMENT_INTERVALS,
            ),
            assumptions=(
                "Unsupported and data-quality states are distinguishable.",
            ),
        ),
        EstimandSpec(
            name=EstimandName.PATH_ATTRIBUTION,
            evidence_phase=EvidencePhase.SILENT_SHADOW,
            target_population="Paired F/S/D outcomes on identical snapshots",
            primary_unit="incident episode and healthy deployment-hour",
            denominator=(
                "Every eligible snapshot evaluated by all frozen policies"
            ),
            inclusion_rule="Policies share identical input and support state",
            adjudication_status="same adjudication used for all paths",
            point_metric="paired_recall_delay_burden_and_fragmentation",
            point_direction=ThresholdDirection.AT_LEAST,
            point_threshold=0.0,
            intent=InferenceIntent.DESCRIPTIVE,
            uncertainty_methods=(
                UncertaintyMethod.DEPLOYMENT_CLUSTER_SENSITIVITY,
            ),
            assumptions=(
                "No path-specific filtering or threshold changes occur.",
            ),
        ),
    ]
    if target_phase in {
        EvidencePhase.OPERATOR_ADVISORY,
        EvidencePhase.OPERATIONAL_CANDIDATE,
    }:
        estimands.append(
            EstimandSpec(
                name=EstimandName.HUMAN_UTILITY,
                evidence_phase=EvidencePhase.OPERATOR_ADVISORY,
                target_population=(
                    "Engineer-incident encounters in the advisory evaluation"
                ),
                primary_unit="engineer-incident encounter",
                denominator=(
                    "All encounters assigned by the signed randomized, paired "
                    "or stepped-rollout workflow design"
                ),
                inclusion_rule=(
                    "Encounter inclusion is independent of favorable outcome"
                ),
                adjudication_status="workflow outcome with override reason",
                point_metric="time_and_scope_relative_improvement",
                point_direction=ThresholdDirection.AT_LEAST,
                point_threshold=0.20,
                intent=InferenceIntent.DESCRIPTIVE,
                uncertainty_methods=(
                    UncertaintyMethod.PAIRED_ROLLOUT_AWARE,
                ),
                assumptions=(
                    "Operators can see NRIM only in the advisory phase.",
                    "Automation bias and override behavior are measured.",
                ),
            )
        )
    return StatisticalContract(
        estimands=tuple(estimands),
        sampling_design=NegativeSamplingDesign(
            design_id="oelr-stratified-review-v1",
            seed_commitment_sha256=(
                sampling_seed_commitment_sha256
            ),
            strata=(
                "deployment",
                "time_of_day",
                "workload",
                "support_state",
            ),
            minimum_inclusion_probability=0.01,
            double_review_rule=(
                "Double-review all severe and unresolved incidents plus a "
                "preregistered prevalence-aware sample."
            ),
        ),
    )


def default_stop_rules() -> tuple[StopRule, ...]:
    values = (
        (
            "OPERATIONAL_WRITE_REACHABLE",
            "An IPTV mutation, ticket, alarm suppression, notification or "
            "remediation path is reachable.",
            True,
        ),
        (
            "IDENTIFIER_OR_FUTURE_LEAKAGE",
            "Direct identity or future outcome information enters inference "
            "or reviewer evidence.",
            True,
        ),
        (
            "EVIDENCE_INTEGRITY_FAILURE",
            "Signature, checkpoint, hash-chain or deterministic replay "
            "verification fails.",
            True,
        ),
        (
            "UNSUPPORTED_SAFE_SEMANTICS_FAILURE",
            "Unsupported or quarantined input becomes HEALTHY or normally "
            "ranked.",
            True,
        ),
        (
            "SYSTEMATIC_DATA_OR_TOPOLOGY_FAILURE",
            "Clock skew, collector outage, schema failure or stale topology "
            "invalidates registered exposure.",
            True,
        ),
        (
            "BLINDING_BREACH",
            "NRIM output is revealed before the initial adjudication outside "
            "the registered exception process.",
            False,
        ),
        (
            "OPERATIONAL_DEGRADATION",
            "The read-only system materially degrades monitoring, network "
            "performance or incident workflow.",
            True,
        ),
        (
            "APPROVAL_WITHDRAWN",
            "A deployment, privacy, security or independent review owner "
            "withdraws approval.",
            False,
        ),
    )
    return tuple(
        StopRule(
            code=code,
            description=description,
            accountable_role="registered_stop_owner",
            automatic_pause=automatic,
            recovery_acceptance=(
                "Preserve evidence, investigate independently, correct under "
                "a new version, and obtain signed recovery acceptance."
            ),
        )
        for code, description, automatic in values
    )


def register_compiled_protocol(
    *,
    compiled: CompiledProtocol,
    registration_path: Path,
    private_key: Ed25519PrivateKey,
    public_key_pem: bytes,
) -> str:
    payload = compiled.model_dump(mode="json")
    envelope = {
        "compiled_protocol": payload,
        "compiled_protocol_sha256": canonical_hash(payload),
        "signature_algorithm": "ed25519",
        "public_key_sha256": bytes_hash(public_key_pem),
        "signature_base64": base64.b64encode(
            private_key.sign(canonical_json(payload).encode("utf-8"))
        ).decode("ascii"),
    }
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        registration_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(canonical_json(envelope))
        stream.flush()
        os.fsync(stream.fileno())
    return envelope["compiled_protocol_sha256"]


def verify_compiled_protocol(
    *,
    registration_path: Path,
    public_key_path: Path,
) -> CompiledProtocol:
    envelope = json.loads(
        registration_path.read_text(encoding="utf-8")
    )
    public_pem = public_key_path.read_bytes()
    if envelope["public_key_sha256"] != bytes_hash(public_pem):
        raise ValueError("Protocol trust anchor differs")
    payload = envelope["compiled_protocol"]
    if envelope["compiled_protocol_sha256"] != canonical_hash(payload):
        raise ValueError("Compiled protocol commitment differs")
    public = serialization.load_pem_public_key(public_pem)
    if not isinstance(public, Ed25519PublicKey):
        raise ValueError("Protocol trust anchor must be Ed25519")
    try:
        public.verify(
            base64.b64decode(
                envelope["signature_base64"],
                validate=True,
            ),
            canonical_json(payload).encode("utf-8"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("Compiled protocol signature is invalid") from error
    compiled = CompiledProtocol.model_validate(payload)
    expected = compile_protocol(compiled.protocol)
    if expected != compiled:
        raise ValueError("Compiled protocol derivation differs")
    return compiled
