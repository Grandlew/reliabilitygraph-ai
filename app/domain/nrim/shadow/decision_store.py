from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Protocol

from pydantic import Field

from .candidate_evidence import StrictDecisionModel
from .decision_envelope import seal_decision_stream
from .decision_events import DecisionEvent, normalize_utc
from .hashing import canonical_json
from .privacy import assert_no_direct_identifiers
from .store import (
    AppendOnlyEvidenceStore,
    ConflictingDuplicateError,
)


class AppendResult(StrictDecisionModel):
    decision_id: str
    inserted_event_count: int = Field(ge=0)
    total_event_count: int = Field(gt=0)
    stream_sha256: str
    inserted: bool


class DecisionStreamRef(StrictDecisionModel):
    decision_id: str
    deployment_pseudonym: str
    decision_cutoff_utc: datetime
    event_count: int = Field(gt=0)
    stream_sha256: str


class DecisionEventStoreProtocol(Protocol):
    def append_stream(
        self,
        events: tuple[DecisionEvent, ...],
    ) -> AppendResult: ...

    def read_stream(self, decision_id: str) -> tuple[DecisionEvent, ...]: ...

    def list_streams(
        self,
        *,
        deployment_pseudonym: str,
        cutoff_start_utc: datetime,
        cutoff_end_utc: datetime,
    ) -> tuple[DecisionStreamRef, ...]: ...


class DecisionEventStore:
    """Decision-event extension over the existing SQLite evidence authority."""

    def __init__(self, evidence_store: AppendOnlyEvidenceStore) -> None:
        self.evidence_store = evidence_store
        self._initialize()

    def _initialize(self) -> None:
        with self.evidence_store._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_events (
                    event_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    deployment_pseudonym TEXT NOT NULL,
                    decision_cutoff_utc TEXT NOT NULL,
                    stream_sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    previous_event_sha256 TEXT NOT NULL,
                    event_sha256 TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(decision_id, stream_sequence)
                );

                CREATE INDEX IF NOT EXISTS decision_events_lookup
                ON decision_events(
                    deployment_pseudonym,
                    decision_cutoff_utc,
                    decision_id,
                    stream_sequence
                );

                CREATE TABLE IF NOT EXISTS decision_stream_commits (
                    stream_sha256 TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    deployment_pseudonym TEXT NOT NULL,
                    decision_cutoff_utc TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    envelope_json TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(decision_id, event_count)
                );

                CREATE TRIGGER IF NOT EXISTS decision_events_no_update
                BEFORE UPDATE ON decision_events
                BEGIN
                    SELECT RAISE(ABORT, 'append-only table');
                END;
                CREATE TRIGGER IF NOT EXISTS decision_events_no_delete
                BEFORE DELETE ON decision_events
                BEGIN
                    SELECT RAISE(ABORT, 'append-only table');
                END;
                CREATE TRIGGER IF NOT EXISTS decision_stream_commits_no_update
                BEFORE UPDATE ON decision_stream_commits
                BEGIN
                    SELECT RAISE(ABORT, 'append-only table');
                END;
                CREATE TRIGGER IF NOT EXISTS decision_stream_commits_no_delete
                BEFORE DELETE ON decision_stream_commits
                BEGIN
                    SELECT RAISE(ABORT, 'append-only table');
                END;
                """
            )

    def append_stream(
        self,
        events: tuple[DecisionEvent, ...],
    ) -> AppendResult:
        envelope = seal_decision_stream(events)
        for event in events:
            assert_no_direct_identifiers(event.model_dump(mode="json"))
        with self.evidence_store._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing_rows = connection.execute(
                    """
                    SELECT payload_json FROM decision_events
                    WHERE decision_id = ? ORDER BY stream_sequence
                    """,
                    (envelope.decision_id,),
                ).fetchall()
                existing = tuple(
                    DecisionEvent.model_validate_json(row["payload_json"])
                    for row in existing_rows
                )
                if len(existing) > len(events):
                    raise ConflictingDuplicateError(
                        "Stored decision stream is longer than supplied stream"
                    )
                if existing != events[: len(existing)]:
                    raise ConflictingDuplicateError(
                        "Decision stream conflicts with immutable history"
                    )
                inserted = events[len(existing) :]
                for event in inserted:
                    connection.execute(
                        """
                        INSERT INTO decision_events (
                            event_id, decision_id, deployment_pseudonym,
                            decision_cutoff_utc, stream_sequence, event_type,
                            recorded_at_utc, previous_event_sha256,
                            event_sha256, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.decision_id,
                            event.deployment_pseudonym,
                            event.decision_cutoff_utc.isoformat(),
                            event.stream_sequence,
                            event.event_type.value,
                            event.recorded_at_utc.isoformat(),
                            event.previous_event_sha256,
                            event.event_sha256,
                            canonical_json(event),
                        ),
                    )
                commit_row = connection.execute(
                    """
                    SELECT stream_sha256 FROM decision_stream_commits
                    WHERE decision_id = ? AND event_count = ?
                    """,
                    (envelope.decision_id, len(events)),
                ).fetchone()
                if commit_row is None:
                    connection.execute(
                        """
                        INSERT INTO decision_stream_commits (
                            stream_sha256, decision_id, deployment_pseudonym,
                            decision_cutoff_utc, event_count, envelope_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            envelope.stream_sha256,
                            envelope.decision_id,
                            envelope.deployment_pseudonym,
                            events[0].decision_cutoff_utc.isoformat(),
                            len(events),
                            canonical_json(envelope),
                        ),
                    )
                elif commit_row["stream_sha256"] != envelope.stream_sha256:
                    raise ConflictingDuplicateError(
                        "Decision stream commit conflicts with stored content"
                    )
                connection.execute("COMMIT")
            except sqlite3.IntegrityError as error:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise ConflictingDuplicateError(
                    "Decision event identifier or sequence conflicts"
                ) from error
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return AppendResult(
            decision_id=envelope.decision_id,
            inserted_event_count=len(inserted),
            total_event_count=len(events),
            stream_sha256=envelope.stream_sha256,
            inserted=bool(inserted),
        )

    def read_stream(self, decision_id: str) -> tuple[DecisionEvent, ...]:
        with self.evidence_store._connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM decision_events
                WHERE decision_id = ? ORDER BY stream_sequence
                """,
                (decision_id,),
            ).fetchall()
        if not rows:
            raise KeyError(decision_id)
        events = tuple(
            DecisionEvent.model_validate_json(row["payload_json"])
            for row in rows
        )
        seal_decision_stream(events)
        return events

    def list_streams(
        self,
        *,
        deployment_pseudonym: str,
        cutoff_start_utc: datetime,
        cutoff_end_utc: datetime,
    ) -> tuple[DecisionStreamRef, ...]:
        start = normalize_utc(cutoff_start_utc)
        end = normalize_utc(cutoff_end_utc)
        if start > end:
            raise ValueError("Decision cutoff range is inverted")
        with self.evidence_store._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.decision_id, c.deployment_pseudonym,
                       c.decision_cutoff_utc, c.event_count, c.stream_sha256
                FROM decision_stream_commits AS c
                JOIN (
                    SELECT decision_id, MAX(event_count) AS event_count
                    FROM decision_stream_commits GROUP BY decision_id
                ) AS latest
                  ON latest.decision_id = c.decision_id
                 AND latest.event_count = c.event_count
                WHERE c.deployment_pseudonym = ?
                  AND c.decision_cutoff_utc >= ?
                  AND c.decision_cutoff_utc <= ?
                ORDER BY c.decision_cutoff_utc, c.decision_id
                """,
                (
                    deployment_pseudonym,
                    start.isoformat(),
                    end.isoformat(),
                ),
            ).fetchall()
        return tuple(
            DecisionStreamRef(
                decision_id=row["decision_id"],
                deployment_pseudonym=row["deployment_pseudonym"],
                decision_cutoff_utc=datetime.fromisoformat(
                    row["decision_cutoff_utc"]
                ),
                event_count=row["event_count"],
                stream_sha256=row["stream_sha256"],
            )
            for row in rows
        )
