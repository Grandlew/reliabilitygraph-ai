from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskState(str, Enum):
    COMPLETE = "complete"
    IMPLEMENTED_NOT_OPERATIONALLY_VALIDATED = (
        "implemented_not_operationally_validated"
    )
    EXTERNAL_EVIDENCE_REQUIRED = "external_evidence_required"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BacklogTaskStatus:
    task_number: int
    name: str
    state: TaskState
    evidence: tuple[str, ...]
    blocker: str | None = None


def historical_replay_gate(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "real_historical_data": metrics.get("real_data") is True,
        "analysis_plan_preregistered": (
            metrics.get("analysis_plan_preregistered") is True
        ),
        "required_features_reconstructable": (
            metrics.get("required_feature_fraction", 0.0) >= 0.95
        ),
        "collector_semantics_accepted": (
            metrics.get("collector_mapping_fraction") == 1.0
            and metrics.get("semantic_mutation_rejection_fraction") == 1.0
        ),
        "incident_alignment": (
            metrics.get("incident_alignment_fraction", 0.0) >= 0.90
        ),
        "label_availability_reported": (
            metrics.get("label_availability_reported") is True
        ),
        "future_leakage_zero": (
            metrics.get("future_leakage_count") == 0
        ),
        "replay_determinism": (
            metrics.get("replay_determinism_fraction") == 1.0
        ),
        "root_cause_mapping": (
            metrics.get("root_cause_mapping_fraction", 0.0) >= 0.95
        ),
        "topology_reconstruction": (
            metrics.get("topology_reconstruction_fraction") == 1.0
        ),
        "no_tuning": (
            metrics.get("model_tuning_event_count") == 0
            and metrics.get("threshold_tuning_event_count") == 0
            and metrics.get("feature_tuning_event_count") == 0
            and metrics.get("support_rule_tuning_event_count") == 0
            and metrics.get("watermark_tuning_event_count") == 0
            and metrics.get("episode_grouping_tuning_event_count") == 0
        ),
        "semantic_deviations_have_lineage": (
            metrics.get("semantic_deviation_count", 0)
            == metrics.get("semantic_lineage_record_count", 0)
        ),
        "data_gap_report_complete": (
            metrics.get("data_gap_report_complete") is True
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "promotion_evidence": False,
        "truth_note": (
            "Historical replay is a readiness check, not prospective "
            "operational efficacy evidence."
        ),
    }


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ReadOnlyBoundaryAttestation(StrictModel):
    attestation_id: str = Field(min_length=1, max_length=128)
    attested_at_utc: datetime
    independent_reviewer_pseudonym: str = Field(
        min_length=12,
        max_length=128,
    )
    credential_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    network_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operational_write_credentials_present: bool
    reachable_mutation_endpoints: tuple[str, ...] = ()
    tested_forbidden_actions: tuple[str, ...] = Field(min_length=5)
    all_forbidden_actions_denied: bool

    @field_validator("attested_at_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Attestation time must be timezone-aware")
        return value.astimezone(timezone.utc)

    @property
    def passed(self) -> bool:
        return (
            not self.operational_write_credentials_present
            and not self.reachable_mutation_endpoints
            and self.all_forbidden_actions_denied
        )


class DressRehearsalEvidence(StrictModel):
    rehearsal_id: str = Field(min_length=1, max_length=128)
    started_at_utc: datetime
    ended_at_utc: datetime
    real_data: bool
    predictions_hidden_from_frontline: bool
    read_only_boundary_passed: bool
    schema_compliance: float = Field(ge=0.0, le=1.0)
    quarantine_attribution_fraction: float = Field(ge=0.0, le=1.0)
    envelope_reproduction_fraction: float = Field(ge=0.0, le=1.0)
    evidence_envelope_completeness: float = Field(ge=0.0, le=1.0)
    evidence_chain_continuity: float = Field(ge=0.0, le=1.0)
    external_checkpoint_success_fraction: float = Field(ge=0.0, le=1.0)
    scheduled_cutoff_completion_fraction: float = Field(ge=0.0, le=1.0)
    registered_availability_floor: float = Field(ge=0.0, le=1.0)
    unsafe_stage2_count: int = Field(ge=0)
    restart_recovery_fraction: float = Field(ge=0.0, le=1.0)
    review_workflow_passed: bool
    emergency_stop_count: int = Field(ge=0)

    @field_validator("started_at_utc", "ended_at_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Rehearsal time must be timezone-aware")
        return value.astimezone(timezone.utc)

    @property
    def consecutive_days(self) -> float:
        return max(
            0.0,
            (self.ended_at_utc - self.started_at_utc).total_seconds()
            / 86400.0,
        )


def dress_rehearsal_gate(
    evidence: DressRehearsalEvidence,
) -> dict[str, Any]:
    criteria = {
        "seven_consecutive_days": evidence.consecutive_days >= 7.0,
        "real_hidden_execution": (
            evidence.real_data
            and evidence.predictions_hidden_from_frontline
        ),
        "read_only": evidence.read_only_boundary_passed,
        "schema": (
            evidence.schema_compliance >= 0.99
            and evidence.quarantine_attribution_fraction == 1.0
        ),
        "reproducibility": evidence.envelope_reproduction_fraction == 1.0,
        "evidence": (
            evidence.evidence_envelope_completeness == 1.0
            and evidence.evidence_chain_continuity == 1.0
            and evidence.external_checkpoint_success_fraction == 1.0
        ),
        "runtime": (
            evidence.scheduled_cutoff_completion_fraction
            >= evidence.registered_availability_floor
        ),
        "support": evidence.unsafe_stage2_count == 0,
        "restart": evidence.restart_recovery_fraction == 1.0,
        "review": evidence.review_workflow_passed,
        "no_stop_condition": evidence.emergency_stop_count == 0,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "consecutive_days": evidence.consecutive_days,
        "promotion_evidence": False,
        "truth_note": (
            "Dress rehearsal validates operations and does not estimate "
            "prospective efficacy."
        ),
    }


STOP_CONDITION_CODES = (
    "OPERATIONAL_WRITE_REACHABLE",
    "IDENTIFIER_OR_FUTURE_LEAKAGE",
    "EVIDENCE_INTEGRITY_FAILURE",
    "UNSUPPORTED_SAFE_SEMANTICS_FAILURE",
    "SYSTEMATIC_DATA_OR_TOPOLOGY_FAILURE",
    "BLINDING_BREACH",
    "OPERATIONAL_DEGRADATION",
    "APPROVAL_WITHDRAWN",
)


def evaluate_stop_conditions(
    observations: Mapping[str, bool],
) -> dict[str, Any]:
    unknown = set(observations) - set(STOP_CONDITION_CODES)
    if unknown:
        raise ValueError(f"Unknown stop conditions: {sorted(unknown)!r}")
    triggered = tuple(
        code for code in STOP_CONDITION_CODES if observations.get(code, False)
    )
    return {
        "pause_required": bool(triggered),
        "triggered": triggered,
        "preserve_evidence": bool(triggered),
        "performance_tuning_authorized": False,
    }


def operational_gate_summary(
    *,
    contracts: Mapping[str, Any],
    replay: Mapping[str, Any],
    evidence_workflow: Mapping[str, Any],
    shadow_runtime: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> dict[str, Any]:
    gates = {
        "contracts_ready": (
            contracts.get("collector_mapping_fraction") == 1.0
            and contracts.get("semantic_mutation_rejection_fraction") == 1.0
            and contracts.get("topology_reconstruction_fraction") == 1.0
            and contracts.get("topology_mutation_rejection_fraction") == 1.0
            and contracts.get("privacy_approved") is True
        ),
        "replay_ready": replay.get("passed") is True,
        "evidence_workflow_ready": (
            evidence_workflow.get("external_anchor_verified") is True
            and evidence_workflow.get("storage_qualification_passed") is True
            and evidence_workflow.get("review_usability_passed") is True
        ),
        "shadow_runtime_ready": (
            shadow_runtime.get("dress_rehearsal_passed") is True
            and shadow_runtime.get("read_only_boundary_passed") is True
        ),
        "pilot_ready": (
            pilot.get("compiled_protocol_signed") is True
            and pilot.get("all_owners_approved") is True
            and pilot.get("deployment_count", 0) >= 5
            and pilot.get("topology_family_count", 0) >= 5
        ),
    }
    return {
        "gates": gates,
        "all_ready": all(gates.values()),
        "truth_note": (
            "A software implementation cannot set externally evidenced fields "
            "to true without their signed artifacts."
        ),
    }
def gate_progress(
    tasks: tuple[BacklogTaskStatus, ...],
) -> dict[str, Any]:
    by_number = {task.task_number: task for task in tasks}

    def complete(numbers: range) -> bool:
        return all(
            by_number.get(number) is not None
            and by_number[number].state is TaskState.COMPLETE
            for number in numbers
        )

    return {
        "gate_A_contracts_ready": complete(range(1, 6)),
        "gate_B_replay_ready": complete(range(6, 11)),
        "gate_C_shadow_runtime_ready": complete(range(11, 16)),
        "gate_D_evidence_workflow_ready": complete(range(16, 19)),
        "gate_E_pilot_ready": complete(range(19, 21)),
    }
