from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EngineerDecision(str, Enum):
    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class OutcomeClassification(str, Enum):
    SOLVED = "solved"
    PARTIALLY_SOLVED = "partially_solved"
    NOT_SOLVED = "not_solved"
    INCONCLUSIVE = "inconclusive"
    FALSE_POSITIVE = "false_positive"


class ReliabilityLearningRecord(BaseModel):
    learning_id: str = Field(
        default_factory=lambda: f"learning_{uuid4().hex}"
    )

    case_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    intervention_id: str | None = None

    engineer_decision: EngineerDecision
    engineer_notes: str = Field(min_length=1)

    outcome: OutcomeClassification
    outcome_evidence_ids: list[str] = Field(default_factory=list)

    reusable_lesson: str | None = None
    approved_for_reuse: bool = False

    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_learning_record(
        self,
    ) -> "ReliabilityLearningRecord":
        if self.outcome in {
            OutcomeClassification.SOLVED,
            OutcomeClassification.PARTIALLY_SOLVED,
        } and not self.outcome_evidence_ids:
            raise ValueError(
                "Outcome evidence is required for solved or partially solved outcomes")

        if self.approved_for_reuse and not self.reusable_lesson:
            raise ValueError(
                "approved_for_reuse requires a non-empty reusable_lesson")

        return self
