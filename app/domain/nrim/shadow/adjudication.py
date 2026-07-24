from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    IncidentAdjudication,
    IncidentSeverity,
)
from .hashing import canonical_hash
from .store import AppendOnlyEvidenceStore


class ReviewerRole(str, Enum):
    REVIEWER = "reviewer"
    RESOLVER = "resolver"
    AUDITOR = "auditor"


class ReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=128)
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    review_window_start_utc: datetime
    review_window_end_utc: datetime
    prediction_ids: tuple[str, ...]
    candidate_source: str = Field(min_length=1, max_length=128)
    requires_double_review: bool
    created_at_utc: datetime


class ReviewWorkflowError(RuntimeError):
    pass


class AdjudicationService:
    """Versioned, blinded-first adjudication over immutable evidence."""

    def __init__(self, store: AppendOnlyEvidenceStore) -> None:
        self.store = store

    @staticmethod
    def _event_id(
        *,
        case_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> str:
        return "review_event_" + canonical_hash(
            {
                "case_id": case_id,
                "event_type": event_type,
                "actor": actor,
                "payload": payload,
            }
        )[:32]

    def create_case(
        self,
        *,
        case_id: str,
        deployment_pseudonym: str,
        review_window_start_utc: datetime,
        review_window_end_utc: datetime,
        prediction_ids: tuple[str, ...],
        candidate_source: str,
        severity_hint: IncidentSeverity | None = None,
        unresolved_hint: bool = False,
        sampled_for_double_review: bool = False,
        actor_pseudonym: str,
    ) -> ReviewCase:
        for prediction_id in prediction_ids:
            envelope = self.store.get_prediction(prediction_id)
            if envelope.deployment_pseudonym != deployment_pseudonym:
                raise ReviewWorkflowError(
                    "Prediction deployment differs from review case"
                )
        requires_double = (
            severity_hint
            in {
                IncidentSeverity.HIGH,
                IncidentSeverity.CRITICAL,
            }
            or unresolved_hint
            or sampled_for_double_review
        )
        case = ReviewCase(
            case_id=case_id,
            deployment_pseudonym=deployment_pseudonym,
            review_window_start_utc=review_window_start_utc,
            review_window_end_utc=review_window_end_utc,
            prediction_ids=prediction_ids,
            candidate_source=candidate_source,
            requires_double_review=requires_double,
            created_at_utc=datetime.now(timezone.utc),
        )
        payload = case.model_dump(mode="json")
        self.store.append_review_event(
            event_id=self._event_id(
                case_id=case_id,
                event_type="case_created",
                actor=actor_pseudonym,
                payload=payload,
            ),
            case_id=case_id,
            event_type="case_created",
            actor_pseudonym=actor_pseudonym,
            payload=payload,
        )
        return case

    def _events(self, case_id: str) -> list[dict[str, Any]]:
        events = self.store.review_events(case_id)
        if not events or events[0]["event_type"] != "case_created":
            raise ReviewWorkflowError("Unknown review case")
        return events

    def case(self, case_id: str) -> ReviewCase:
        events = self._events(case_id)
        return ReviewCase.model_validate(events[0]["payload"])

    def submit_initial(
        self,
        *,
        case_id: str,
        adjudication: IncidentAdjudication,
    ) -> None:
        case = self.case(case_id)
        if not adjudication.blinded_initial_assessment:
            raise ReviewWorkflowError("Initial review must be marked blinded")
        if adjudication.deployment_pseudonym != case.deployment_pseudonym:
            raise ReviewWorkflowError("Adjudication deployment differs")
        existing = self._events(case_id)
        if any(
            item["event_type"] == "initial_submitted"
            and item["actor_pseudonym"]
            == adjudication.reviewer_id_pseudonym
            for item in existing
        ):
            raise ReviewWorkflowError("Reviewer already submitted initial review")
        self.store.append_adjudication(adjudication)
        payload = {
            "adjudication_id": adjudication.adjudication_id,
            "version": adjudication.adjudication_version,
        }
        self.store.append_review_event(
            event_id=self._event_id(
                case_id=case_id,
                event_type="initial_submitted",
                actor=adjudication.reviewer_id_pseudonym,
                payload=payload,
            ),
            case_id=case_id,
            event_type="initial_submitted",
            actor_pseudonym=adjudication.reviewer_id_pseudonym,
            payload=payload,
        )

    def reveal_nrim(
        self,
        *,
        case_id: str,
        reviewer_pseudonym: str,
    ) -> tuple[dict[str, Any], ...]:
        case = self.case(case_id)
        events = self._events(case_id)
        if not any(
            item["event_type"] == "initial_submitted"
            and item["actor_pseudonym"] == reviewer_pseudonym
            for item in events
        ):
            raise ReviewWorkflowError(
                "NRIM evidence cannot be revealed before initial review"
            )
        payload = {"prediction_ids": list(case.prediction_ids)}
        self.store.append_review_event(
            event_id=self._event_id(
                case_id=case_id,
                event_type="nrim_revealed",
                actor=reviewer_pseudonym,
                payload=payload,
            ),
            case_id=case_id,
            event_type="nrim_revealed",
            actor_pseudonym=reviewer_pseudonym,
            payload=payload,
        )
        return tuple(
            self.store.get_prediction(item).model_dump(mode="json")
            for item in case.prediction_ids
        )

    def submit_post_reveal(
        self,
        *,
        case_id: str,
        adjudication: IncidentAdjudication,
    ) -> None:
        events = self._events(case_id)
        reviewer = adjudication.reviewer_id_pseudonym
        if adjudication.blinded_initial_assessment:
            raise ReviewWorkflowError(
                "Post-reveal adjudication cannot be marked blinded"
            )
        if not any(
            item["event_type"] == "nrim_revealed"
            and item["actor_pseudonym"] == reviewer
            for item in events
        ):
            raise ReviewWorkflowError("Reviewer has not revealed NRIM evidence")
        versions = self.store.list_adjudications(
            adjudication_id=adjudication.adjudication_id
        )
        if not versions:
            raise ReviewWorkflowError(
                "Post-reveal version requires preserved initial version"
            )
        if adjudication.adjudication_version != (
            versions[-1].adjudication_version + 1
        ):
            raise ReviewWorkflowError(
                "Adjudication versions must be consecutive"
            )
        self.store.append_adjudication(adjudication)
        payload = {
            "adjudication_id": adjudication.adjudication_id,
            "version": adjudication.adjudication_version,
        }
        self.store.append_review_event(
            event_id=self._event_id(
                case_id=case_id,
                event_type="post_reveal_submitted",
                actor=reviewer,
                payload=payload,
            ),
            case_id=case_id,
            event_type="post_reveal_submitted",
            actor_pseudonym=reviewer,
            payload=payload,
        )

    def record_resolution(
        self,
        *,
        case_id: str,
        resolver_pseudonym: str,
        final_adjudication: IncidentAdjudication,
        disagreement_reason: str,
    ) -> None:
        case = self.case(case_id)
        initial_reviewers = {
            item["actor_pseudonym"]
            for item in self._events(case_id)
            if item["event_type"] == "initial_submitted"
        }
        if len(initial_reviewers) < 2 and case.requires_double_review:
            raise ReviewWorkflowError(
                "Required second review has not been completed"
            )
        if not disagreement_reason.strip():
            raise ReviewWorkflowError("Resolution reason is required")
        self.store.append_adjudication(final_adjudication)
        payload = {
            "adjudication_id": final_adjudication.adjudication_id,
            "version": final_adjudication.adjudication_version,
            "disagreement_reason": disagreement_reason,
            "preserved_reviewer_count": len(initial_reviewers),
        }
        self.store.append_review_event(
            event_id=self._event_id(
                case_id=case_id,
                event_type="disagreement_resolved",
                actor=resolver_pseudonym,
                payload=payload,
            ),
            case_id=case_id,
            event_type="disagreement_resolved",
            actor_pseudonym=resolver_pseudonym,
            payload=payload,
        )

    def workflow_status(self, case_id: str) -> dict[str, Any]:
        case = self.case(case_id)
        events = self._events(case_id)
        reviewers = {
            item["actor_pseudonym"]
            for item in events
            if item["event_type"] == "initial_submitted"
        }
        return {
            "case_id": case_id,
            "initial_review_count": len(reviewers),
            "requires_double_review": case.requires_double_review,
            "double_review_complete": (
                not case.requires_double_review or len(reviewers) >= 2
            ),
            "revealed_reviewer_count": len(
                {
                    item["actor_pseudonym"]
                    for item in events
                    if item["event_type"] == "nrim_revealed"
                }
            ),
            "resolved": any(
                item["event_type"] == "disagreement_resolved"
                for item in events
            ),
        }
