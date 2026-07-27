from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import Field

from .candidate_evidence import StrictDecisionModel
from .decision_projection import (
    DecisionComparison,
    DecisionSnapshot,
    project_decision,
)
from .decision_store import DecisionEventStore, DecisionStreamRef
from .hashing import bytes_hash, canonical_json, canonical_jsonl, typed_hash
from .privacy import redacted_structure


class DecisionSummary(StrictDecisionModel):
    decision_id: str
    deployment_pseudonym: str
    decision_cutoff_utc: datetime
    final_state: str
    support_state: str
    data_quality_state: str
    activation_path: str
    stream_sha256: str


class DecisionPage(StrictDecisionModel):
    items: tuple[DecisionSummary, ...]
    next_cursor: str | None
    schema_version: str = "0.7.2"


def _snapshot(store: DecisionEventStore, decision_id: str, view: str, at):
    try:
        events = store.read_stream(decision_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Decision not found") from error
    try:
        return project_decision(
            events,
            view=view,
            as_known_at_utc=at,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _summary(
    store: DecisionEventStore,
    reference: DecisionStreamRef,
) -> DecisionSummary:
    snapshot = project_decision(
        store.read_stream(reference.decision_id),
        view="latest",
    )
    assert isinstance(snapshot, DecisionSnapshot)
    return DecisionSummary(
        decision_id=snapshot.decision_id,
        deployment_pseudonym=snapshot.deployment_pseudonym,
        decision_cutoff_utc=snapshot.decision_cutoff_utc,
        final_state=snapshot.final_state.value,
        support_state=snapshot.support_state.value,
        data_quality_state=snapshot.data_quality_state.value,
        activation_path=snapshot.activation_path,
        stream_sha256=reference.stream_sha256,
    )


def create_decision_router(*, store: DecisionEventStore) -> APIRouter:
    router = APIRouter(prefix="/shadow/decisions", tags=["decision-evidence"])

    @router.get("")
    def list_decisions(
        deployment_pseudonym: str,
        cutoff_start_utc: datetime,
        cutoff_end_utc: datetime,
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> DecisionPage:
        references = store.list_streams(
            deployment_pseudonym=deployment_pseudonym,
            cutoff_start_utc=cutoff_start_utc,
            cutoff_end_utc=cutoff_end_utc,
        )
        if cursor is not None:
            references = tuple(
                item for item in references if item.decision_id > cursor
            )
        selected = references[:limit]
        next_cursor = (
            selected[-1].decision_id
            if len(references) > len(selected)
            else None
        )
        return DecisionPage(
            items=tuple(_summary(store, item) for item in selected),
            next_cursor=next_cursor,
        )

    @router.get("/stream")
    def stream_decisions(
        deployment_pseudonym: str,
        cutoff_start_utc: datetime,
        cutoff_end_utc: datetime,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> Response:
        references = store.list_streams(
            deployment_pseudonym=deployment_pseudonym,
            cutoff_start_utc=cutoff_start_utc,
            cutoff_end_utc=cutoff_end_utc,
        )
        if last_event_id is not None:
            references = tuple(
                item
                for item in references
                if item.decision_id > last_event_id
            )
        lines: list[str] = ["retry: 5000\n"]
        for reference in references:
            summary = _summary(store, reference)
            lines.extend(
                [
                    f"id: {summary.decision_id}\n",
                    "event: decision\n",
                    f"data: {canonical_json(summary)}\n\n",
                ]
            )
        lines.append(": heartbeat\n\n")
        return Response(
            "".join(lines).encode("utf-8"),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/{decision_id}/events")
    def get_events(
        decision_id: str,
        redact: bool = False,
    ) -> list[dict]:
        try:
            events = store.read_stream(decision_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="Decision not found",
            ) from error
        if not redact:
            return [item.model_dump(mode="json") for item in events]
        return [
            {
                **item.model_dump(mode="json", exclude={"payload"}),
                "payload": redacted_structure(
                    item.payload.model_dump(mode="json")
                ),
                "redacted": True,
            }
            for item in events
        ]

    @router.get("/{decision_id}/evidence")
    def get_evidence(decision_id: str) -> dict:
        snapshot = _snapshot(store, decision_id, "latest", None)
        assert isinstance(snapshot, DecisionSnapshot)
        return {
            "decision_id": snapshot.decision_id,
            "final_state": snapshot.final_state,
            "support_state": snapshot.support_state,
            "data_quality_state": snapshot.data_quality_state,
            "candidate_set": snapshot.candidate_set,
            "ranking": snapshot.ranking,
            "causal_evidence": snapshot.causal_evidence,
            "stage2_exposed": bool(snapshot.ranking),
        }

    @router.get("/{decision_id}/export")
    def export_decision(decision_id: str) -> Response:
        try:
            events = store.read_stream(decision_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="Decision not found",
            ) from error
        body = canonical_jsonl(events)
        manifest = {
            "decision_id": decision_id,
            "event_count": len(events),
            "jsonl_sha256": bytes_hash(body),
            "schema_version": "0.7.2",
        }
        return Response(
            body,
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": (
                    f'attachment; filename="{decision_id}.jsonl"'
                ),
                "X-Decision-Manifest-SHA256": typed_hash(
                    type_name="decision_export_manifest",
                    schema_version="0.7.2",
                    value=manifest,
                ),
            },
        )

    @router.get("/{decision_id}")
    def get_decision(
        decision_id: str,
        view: Literal[
            "original",
            "as_known_at",
            "latest",
            "comparison",
        ] = "latest",
        as_known_at_utc: datetime | None = None,
    ) -> DecisionSnapshot | DecisionComparison:
        return _snapshot(store, decision_id, view, as_known_at_utc)

    return router
