from __future__ import annotations

from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.nrim.shadow.decision_api import create_decision_router
from app.domain.nrim.shadow.decision_store import DecisionEventStore
from app.domain.nrim.shadow.store import AppendOnlyEvidenceStore


def _client(tmp_path, decision_events):
    store = DecisionEventStore(
        AppendOnlyEvidenceStore(tmp_path / "api.sqlite")
    )
    store.append_stream(decision_events)
    app = FastAPI()
    app.include_router(create_decision_router(store=store))
    return TestClient(app), app, store


def _list_params(decision_events) -> dict[str, str]:
    cutoff = decision_events[0].decision_cutoff_utc
    return {
        "deployment_pseudonym": decision_events[0].deployment_pseudonym,
        "cutoff_start_utc": (cutoff - timedelta(hours=1)).isoformat(),
        "cutoff_end_utc": (cutoff + timedelta(hours=1)).isoformat(),
    }


def test_decision_router_openapi_is_strictly_read_only(
    tmp_path,
    decision_events,
) -> None:
    _, app, _ = _client(tmp_path, decision_events)
    for path, operations in app.openapi()["paths"].items():
        if path.startswith("/shadow/decisions"):
            assert set(operations).issubset({"get", "head", "options"})


def test_list_returns_deterministic_page(tmp_path, decision_events) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    first = client.get("/shadow/decisions", params=_list_params(decision_events))
    second = client.get("/shadow/decisions", params=_list_params(decision_events))
    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["next_cursor"] is None


def test_detail_supports_original_and_latest_views(
    tmp_path,
    decision_events,
) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    decision_id = decision_events[0].decision_id
    original = client.get(
        f"/shadow/decisions/{decision_id}",
        params={"view": "original"},
    )
    latest = client.get(
        f"/shadow/decisions/{decision_id}",
        params={"view": "latest"},
    )
    assert original.status_code == latest.status_code == 200
    assert original.json()["prediction"] == latest.json()["prediction"]


def test_detail_returns_safe_not_found(tmp_path, decision_events) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    response = client.get("/shadow/decisions/decision_missing_12345678")
    assert response.status_code == 404
    assert response.json() == {"detail": "Decision not found"}


def test_event_view_retains_hashes_when_redacted(
    tmp_path,
    decision_events,
) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    decision_id = decision_events[0].decision_id
    response = client.get(
        f"/shadow/decisions/{decision_id}/events",
        params={"redact": "true"},
    )
    assert response.status_code == 200
    assert response.json()[0]["event_sha256"] == (
        decision_events[0].event_sha256
    )
    assert response.json()[0]["redacted"] is True


def test_evidence_view_separates_score_and_obligations(
    tmp_path,
    decision_events,
) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    decision_id = decision_events[0].decision_id
    payload = client.get(
        f"/shadow/decisions/{decision_id}/evidence"
    ).json()
    assert payload["ranking"]
    assert payload["causal_evidence"]["obligations"]
    assert "causal_proof" not in payload["causal_evidence"]


def test_export_is_canonical_and_digest_bound(
    tmp_path,
    decision_events,
) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    decision_id = decision_events[0].decision_id
    first = client.get(f"/shadow/decisions/{decision_id}/export")
    second = client.get(f"/shadow/decisions/{decision_id}/export")
    assert first.content == second.content
    assert first.content.endswith(b"\n")
    assert first.headers["x-decision-manifest-sha256"]


def test_sse_supports_last_event_id_resume(
    tmp_path,
    decision_events,
) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    params = _list_params(decision_events)
    first = client.get("/shadow/decisions/stream", params=params)
    assert first.status_code == 200
    assert f"id: {decision_events[0].decision_id}" in first.text
    resumed = client.get(
        "/shadow/decisions/stream",
        params=params,
        headers={"Last-Event-ID": decision_events[0].decision_id},
    )
    assert "event: decision" not in resumed.text
    assert ": heartbeat" in resumed.text


def test_reads_do_not_create_new_evidence(tmp_path, decision_events) -> None:
    client, _, store = _client(tmp_path, decision_events)
    before = store.read_stream(decision_events[0].decision_id)
    client.get(f"/shadow/decisions/{decision_events[0].decision_id}")
    after = store.read_stream(decision_events[0].decision_id)
    assert after == before


def test_mutation_methods_are_unreachable(tmp_path, decision_events) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    decision_id = decision_events[0].decision_id
    for method in ("post", "put", "patch", "delete"):
        response = client.request(
            method.upper(),
            f"/shadow/decisions/{decision_id}",
            json={},
        )
        assert response.status_code == 405


def test_invalid_as_known_request_fails_closed(
    tmp_path,
    decision_events,
) -> None:
    client, _, _ = _client(tmp_path, decision_events)
    response = client.get(
        f"/shadow/decisions/{decision_events[0].decision_id}",
        params={"view": "as_known_at"},
    )
    assert response.status_code == 422
