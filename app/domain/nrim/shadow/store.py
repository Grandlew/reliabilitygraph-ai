from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .contracts import (
    IncidentAdjudication,
    OperationalEvent,
    PredictionEnvelope,
    ServingSnapshot,
    SnapshotMode,
    TelemetryObservation,
    DeploymentProfile,
    TopologyComponent,
    TopologyEdge,
)
from .hashing import canonical_hash, canonical_json


class AppendOnlyViolation(RuntimeError):
    pass


class ConflictingDuplicateError(AppendOnlyViolation):
    pass


class MissingSnapshotError(AppendOnlyViolation):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppendOnlyEvidenceStore:
    """Transactional local evidence store with database-enforced immutability.

    SQLite is the deterministic pilot/test adapter. The schema deliberately
    uses append-only primitives that map directly to the recommended
    PostgreSQL production repository.
    """

    _IMMUTABLE_TABLES = (
        "telemetry",
        "operational_events",
        "topology_components",
        "topology_edges",
        "deployment_profiles",
        "quarantine",
        "snapshots",
        "predictions",
        "state_transitions",
        "adjudications",
        "review_events",
        "audit_events",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = FULL;

                CREATE TABLE IF NOT EXISTS telemetry (
                    collector_id TEXT NOT NULL,
                    source_sequence_id TEXT NOT NULL,
                    deployment_pseudonym TEXT NOT NULL,
                    event_time_utc TEXT NOT NULL,
                    ingestion_time_utc TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL,
                    PRIMARY KEY (collector_id, source_sequence_id)
                );

                CREATE TABLE IF NOT EXISTS operational_events (
                    source_sequence_id TEXT PRIMARY KEY,
                    deployment_pseudonym TEXT NOT NULL,
                    event_time_utc TEXT NOT NULL,
                    ingestion_time_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS topology_components (
                    record_id TEXT PRIMARY KEY,
                    deployment_pseudonym TEXT NOT NULL,
                    topology_version TEXT NOT NULL,
                    valid_from_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS topology_edges (
                    record_id TEXT PRIMARY KEY,
                    deployment_pseudonym TEXT NOT NULL,
                    topology_version TEXT NOT NULL,
                    valid_from_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deployment_profiles (
                    record_id TEXT PRIMARY KEY,
                    deployment_pseudonym TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    valid_from_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quarantine (
                    quarantine_id TEXT PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    safe_structure_json TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_hash TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    deployment_pseudonym TEXT NOT NULL,
                    decision_cutoff_utc TEXT NOT NULL,
                    snapshot_mode TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    snapshot_hash TEXT NOT NULL,
                    deployment_pseudonym TEXT NOT NULL,
                    decision_cutoff_utc TEXT NOT NULL,
                    snapshot_mode TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    previous_chain_hash TEXT NOT NULL,
                    chain_hash TEXT NOT NULL UNIQUE,
                    appended_at_utc TEXT NOT NULL,
                    FOREIGN KEY (snapshot_hash)
                        REFERENCES snapshots(snapshot_hash)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS
                    one_prospective_prediction_per_cutoff
                ON predictions(deployment_pseudonym, decision_cutoff_utc)
                WHERE snapshot_mode = 'prospective';

                CREATE TABLE IF NOT EXISTS state_transitions (
                    transition_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL,
                    deployment_pseudonym TEXT NOT NULL,
                    decision_cutoff_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL,
                    FOREIGN KEY (prediction_id)
                        REFERENCES predictions(prediction_id)
                );

                CREATE TABLE IF NOT EXISTS adjudications (
                    adjudication_id TEXT NOT NULL,
                    adjudication_version INTEGER NOT NULL,
                    deployment_pseudonym TEXT NOT NULL,
                    reviewer_id_pseudonym TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL,
                    PRIMARY KEY (adjudication_id, adjudication_version)
                );

                CREATE TABLE IF NOT EXISTS review_events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_pseudonym TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    actor_pseudonym TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    appended_at_utc TEXT NOT NULL
                );
                """
            )
            for table in self._IMMUTABLE_TABLES:
                connection.executescript(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, 'append-only table');
                    END;
                    CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, 'append-only table');
                    END;
                    """
                )

    @staticmethod
    def _insert_idempotent(
        connection: sqlite3.Connection,
        *,
        query: str,
        parameters: tuple[Any, ...],
        duplicate_query: str,
        duplicate_parameters: tuple[Any, ...],
        payload_hash: str,
    ) -> bool:
        try:
            connection.execute(query, parameters)
            return True
        except sqlite3.IntegrityError:
            row = connection.execute(
                duplicate_query,
                duplicate_parameters,
            ).fetchone()
            if row is not None and row["payload_hash"] == payload_hash:
                return False
            raise ConflictingDuplicateError(
                "Identifier was reused for different immutable content"
            ) from None

    def append_telemetry(self, value: TelemetryObservation) -> bool:
        payload = canonical_json(value)
        payload_hash = canonical_hash(value)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                inserted = self._insert_idempotent(
                    connection,
                    query=(
                        "INSERT INTO telemetry VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    parameters=(
                        value.collector_id,
                        value.source_sequence_id,
                        value.deployment_pseudonym,
                        value.event_time_utc.isoformat(),
                        value.ingestion_time_utc.isoformat(),
                        value.metric_name.value,
                        payload,
                        payload_hash,
                        _now(),
                    ),
                    duplicate_query=(
                        "SELECT payload_hash FROM telemetry "
                        "WHERE collector_id = ? AND source_sequence_id = ?"
                    ),
                    duplicate_parameters=(
                        value.collector_id,
                        value.source_sequence_id,
                    ),
                    payload_hash=payload_hash,
                )
                connection.execute("COMMIT")
                return inserted
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def append_operational_event(self, value: OperationalEvent) -> bool:
        payload = canonical_json(value)
        payload_hash = canonical_hash(value)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                inserted = self._insert_idempotent(
                    connection,
                    query=(
                        "INSERT INTO operational_events VALUES "
                        "(?, ?, ?, ?, ?, ?, ?)"
                    ),
                    parameters=(
                        value.source_sequence_id,
                        value.deployment_pseudonym,
                        value.event_time_utc.isoformat(),
                        value.ingestion_time_utc.isoformat(),
                        payload,
                        payload_hash,
                        _now(),
                    ),
                    duplicate_query=(
                        "SELECT payload_hash FROM operational_events "
                        "WHERE source_sequence_id = ?"
                    ),
                    duplicate_parameters=(value.source_sequence_id,),
                    payload_hash=payload_hash,
                )
                connection.execute("COMMIT")
                return inserted
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _append_versioned_contract(
        self,
        *,
        table: str,
        record_id: str,
        deployment_pseudonym: str,
        version: str,
        valid_from_utc: datetime,
        value: Any,
        version_column: str,
    ) -> bool:
        if table not in {
            "topology_components",
            "topology_edges",
            "deployment_profiles",
        }:
            raise ValueError("Unsupported versioned contract table")
        payload = canonical_json(value)
        payload_hash = canonical_hash(value)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                inserted = self._insert_idempotent(
                    connection,
                    query=(
                        f"INSERT INTO {table} "
                        f"(record_id, deployment_pseudonym, {version_column}, "
                        "valid_from_utc, payload_json, payload_hash, "
                        "appended_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?)"
                    ),
                    parameters=(
                        record_id,
                        deployment_pseudonym,
                        version,
                        valid_from_utc.isoformat(),
                        payload,
                        payload_hash,
                        _now(),
                    ),
                    duplicate_query=(
                        f"SELECT payload_hash FROM {table} "
                        "WHERE record_id = ?"
                    ),
                    duplicate_parameters=(record_id,),
                    payload_hash=payload_hash,
                )
                connection.execute("COMMIT")
                return inserted
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def append_topology_component(
        self,
        value: TopologyComponent,
    ) -> bool:
        record_id = "topology_component_" + canonical_hash(
            {
                "deployment": value.deployment_pseudonym,
                "component": value.component_pseudonym,
                "version": value.topology_version,
                "valid_from": value.valid_from_utc,
            }
        )[:32]
        return self._append_versioned_contract(
            table="topology_components",
            record_id=record_id,
            deployment_pseudonym=value.deployment_pseudonym,
            version=value.topology_version,
            valid_from_utc=value.valid_from_utc,
            value=value,
            version_column="topology_version",
        )

    def append_topology_edge(self, value: TopologyEdge) -> bool:
        record_id = "topology_edge_" + canonical_hash(
            {
                "deployment": value.deployment_pseudonym,
                "source": value.source_component,
                "destination": value.destination_component,
                "type": value.dependency_type,
                "version": value.topology_version,
                "valid_from": value.valid_from_utc,
            }
        )[:32]
        return self._append_versioned_contract(
            table="topology_edges",
            record_id=record_id,
            deployment_pseudonym=value.deployment_pseudonym,
            version=value.topology_version,
            valid_from_utc=value.valid_from_utc,
            value=value,
            version_column="topology_version",
        )

    def append_deployment_profile(
        self,
        value: DeploymentProfile,
    ) -> bool:
        record_id = "deployment_profile_" + canonical_hash(
            {
                "deployment": value.deployment_pseudonym,
                "version": value.profile_version,
                "valid_from": value.valid_from_utc,
            }
        )[:32]
        return self._append_versioned_contract(
            table="deployment_profiles",
            record_id=record_id,
            deployment_pseudonym=value.deployment_pseudonym,
            version=value.profile_version,
            valid_from_utc=value.valid_from_utc,
            value=value,
            version_column="profile_version",
        )

    def topology_contracts(
        self,
        *,
        deployment_pseudonym: str | None = None,
    ) -> tuple[
        list[TopologyComponent],
        list[TopologyEdge],
        list[DeploymentProfile],
    ]:
        def rows(table: str) -> list[sqlite3.Row]:
            query = f"SELECT payload_json FROM {table}"
            parameters: tuple[Any, ...] = ()
            if deployment_pseudonym is not None:
                query += " WHERE deployment_pseudonym = ?"
                parameters = (deployment_pseudonym,)
            query += " ORDER BY valid_from_utc, record_id"
            with self._connection() as connection:
                return connection.execute(query, parameters).fetchall()

        return (
            [
                TopologyComponent.model_validate_json(item["payload_json"])
                for item in rows("topology_components")
            ],
            [
                TopologyEdge.model_validate_json(item["payload_json"])
                for item in rows("topology_edges")
            ],
            [
                DeploymentProfile.model_validate_json(item["payload_json"])
                for item in rows("deployment_profiles")
            ],
        )

    def append_quarantine(
        self,
        *,
        record_type: str,
        original_payload_hash: str,
        safe_structure: Any,
        reasons: tuple[str, ...],
    ) -> str:
        payload = {
            "record_type": record_type,
            "payload_hash": original_payload_hash,
            "safe_structure": safe_structure,
            "reasons": reasons,
        }
        quarantine_id = "quarantine_" + canonical_hash(payload)[:32]
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO quarantine VALUES
                (?, ?, ?, ?, ?, ?)
                """,
                (
                    quarantine_id,
                    record_type,
                    original_payload_hash,
                    canonical_json(safe_structure),
                    canonical_json(reasons),
                    _now(),
                ),
            )
        return quarantine_id

    def telemetry_between(
        self,
        *,
        deployment_pseudonym: str,
        event_start_utc: datetime,
        event_end_utc: datetime,
        ingested_by_utc: datetime,
    ) -> list[TelemetryObservation]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM telemetry
                WHERE deployment_pseudonym = ?
                  AND event_time_utc >= ?
                  AND event_time_utc <= ?
                  AND ingestion_time_utc <= ?
                ORDER BY event_time_utc, collector_id, source_sequence_id
                """,
                (
                    deployment_pseudonym,
                    event_start_utc.isoformat(),
                    event_end_utc.isoformat(),
                    ingested_by_utc.isoformat(),
                ),
            ).fetchall()
        return [
            TelemetryObservation.model_validate_json(row["payload_json"])
            for row in rows
        ]

    def operational_events_between(
        self,
        *,
        deployment_pseudonym: str,
        event_start_utc: datetime,
        event_end_utc: datetime,
        ingested_by_utc: datetime,
    ) -> list[OperationalEvent]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM operational_events
                WHERE deployment_pseudonym = ?
                  AND event_time_utc >= ?
                  AND event_time_utc <= ?
                  AND ingestion_time_utc <= ?
                ORDER BY event_time_utc, source_sequence_id
                """,
                (
                    deployment_pseudonym,
                    event_start_utc.isoformat(),
                    event_end_utc.isoformat(),
                    ingested_by_utc.isoformat(),
                ),
            ).fetchall()
        return [
            OperationalEvent.model_validate_json(row["payload_json"])
            for row in rows
        ]

    def append_snapshot(self, snapshot: ServingSnapshot) -> tuple[str, bool]:
        snapshot_hash = snapshot.content_hash()
        payload = canonical_json(snapshot)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO snapshots VALUES
                    (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_hash,
                        snapshot.snapshot_id,
                        snapshot.deployment_pseudonym,
                        snapshot.decision_cutoff_utc.isoformat(),
                        snapshot.mode.value,
                        payload,
                        _now(),
                    ),
                )
                inserted = cursor.rowcount == 1
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return snapshot_hash, inserted

    def list_snapshots(
        self,
        *,
        deployment_pseudonym: str,
        mode: SnapshotMode = SnapshotMode.PROSPECTIVE,
        through_cutoff_utc: datetime | None = None,
    ) -> list[ServingSnapshot]:
        query = (
            "SELECT payload_json FROM snapshots "
            "WHERE deployment_pseudonym = ? AND snapshot_mode = ?"
        )
        parameters: list[Any] = [deployment_pseudonym, mode.value]
        if through_cutoff_utc is not None:
            query += " AND decision_cutoff_utc <= ?"
            parameters.append(through_cutoff_utc.isoformat())
        query += " ORDER BY decision_cutoff_utc, snapshot_hash"
        with self._connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            ServingSnapshot.model_validate_json(row["payload_json"])
            for row in rows
        ]

    def append_prediction(
        self,
        *,
        envelope: PredictionEnvelope,
        snapshot_hash: str,
    ) -> tuple[str, bool]:
        payload = canonical_json(envelope)
        payload_hash = canonical_hash(envelope)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                snapshot = connection.execute(
                    "SELECT 1 FROM snapshots WHERE snapshot_hash = ?",
                    (snapshot_hash,),
                ).fetchone()
                if snapshot is None:
                    raise MissingSnapshotError(
                        "Prediction references an unknown snapshot"
                    )
                previous = connection.execute(
                    """
                    SELECT chain_hash FROM predictions
                    ORDER BY rowid DESC LIMIT 1
                    """
                ).fetchone()
                previous_hash = (
                    previous["chain_hash"] if previous is not None else "0" * 64
                )
                chain_hash = hashlib.sha256(
                    f"{previous_hash}:{payload_hash}".encode("ascii")
                ).hexdigest()
                try:
                    connection.execute(
                        """
                        INSERT INTO predictions VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            envelope.prediction_id,
                            snapshot_hash,
                            envelope.deployment_pseudonym,
                            envelope.decision_cutoff_utc.isoformat(),
                            envelope.snapshot_mode.value,
                            payload,
                            payload_hash,
                            previous_hash,
                            chain_hash,
                            _now(),
                        ),
                    )
                except sqlite3.IntegrityError:
                    row = connection.execute(
                        """
                        SELECT payload_hash, chain_hash FROM predictions
                        WHERE prediction_id = ?
                        """,
                        (envelope.prediction_id,),
                    ).fetchone()
                    if row is None or row["payload_hash"] != payload_hash:
                        raise ConflictingDuplicateError(
                            "Prediction/cutoff already has different evidence"
                        ) from None
                    connection.execute("ROLLBACK")
                    return row["chain_hash"], False
                connection.execute("COMMIT")
                return chain_hash, True
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def append_state_transition(
        self,
        *,
        prediction_id: str,
        deployment_pseudonym: str,
        decision_cutoff_utc: datetime,
        payload: dict[str, Any],
    ) -> str:
        payload_hash = canonical_hash(payload)
        transition_id = "transition_" + canonical_hash(
            {
                "prediction_id": prediction_id,
                "payload_hash": payload_hash,
            }
        )[:32]
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO state_transitions VALUES
                (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition_id,
                    prediction_id,
                    deployment_pseudonym,
                    decision_cutoff_utc.isoformat(),
                    canonical_json(payload),
                    payload_hash,
                    _now(),
                ),
            )
        return transition_id

    def get_prediction(self, prediction_id: str) -> PredictionEnvelope:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        if row is None:
            raise KeyError(prediction_id)
        return PredictionEnvelope.model_validate_json(row["payload_json"])

    def list_predictions(self) -> list[PredictionEnvelope]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM predictions ORDER BY rowid"
            ).fetchall()
        return [
            PredictionEnvelope.model_validate_json(row["payload_json"])
            for row in rows
        ]

    def append_adjudication(self, value: IncidentAdjudication) -> bool:
        payload = canonical_json(value)
        payload_hash = canonical_hash(value)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO adjudications VALUES
                (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    value.adjudication_id,
                    value.adjudication_version,
                    value.deployment_pseudonym,
                    value.reviewer_id_pseudonym,
                    payload,
                    payload_hash,
                    _now(),
                ),
            )
            if cursor.rowcount == 1:
                return True
            row = connection.execute(
                """
                SELECT payload_hash FROM adjudications
                WHERE adjudication_id = ? AND adjudication_version = ?
                """,
                (value.adjudication_id, value.adjudication_version),
            ).fetchone()
        if row is None or row["payload_hash"] != payload_hash:
            raise ConflictingDuplicateError(
                "Adjudication version already contains different content"
            )
        return False

    def list_adjudications(
        self,
        *,
        adjudication_id: str | None = None,
    ) -> list[IncidentAdjudication]:
        query = "SELECT payload_json FROM adjudications"
        parameters: tuple[Any, ...] = ()
        if adjudication_id is not None:
            query += " WHERE adjudication_id = ?"
            parameters = (adjudication_id,)
        query += " ORDER BY adjudication_id, adjudication_version"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            IncidentAdjudication.model_validate_json(row["payload_json"])
            for row in rows
        ]

    def append_review_event(
        self,
        *,
        event_id: str,
        case_id: str,
        event_type: str,
        actor_pseudonym: str,
        payload: dict[str, Any],
    ) -> bool:
        payload_hash = canonical_hash(payload)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO review_events VALUES
                (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    case_id,
                    event_type,
                    actor_pseudonym,
                    canonical_json(payload),
                    payload_hash,
                    _now(),
                ),
            )
        return cursor.rowcount == 1

    def review_events(self, case_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT event_id, event_type, actor_pseudonym, payload_json,
                       appended_at_utc
                FROM review_events WHERE case_id = ? ORDER BY rowid
                """,
                (case_id,),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "actor_pseudonym": row["actor_pseudonym"],
                "payload": json.loads(row["payload_json"]),
                "appended_at_utc": row["appended_at_utc"],
            }
            for row in rows
        ]

    def verify_prediction_chain(self) -> bool:
        previous = "0" * 64
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_hash, previous_chain_hash, chain_hash
                FROM predictions ORDER BY rowid
                """
            ).fetchall()
        for row in rows:
            if row["previous_chain_hash"] != previous:
                return False
            expected = hashlib.sha256(
                f"{previous}:{row['payload_hash']}".encode("ascii")
            ).hexdigest()
            if row["chain_hash"] != expected:
                return False
            previous = expected
        return True
