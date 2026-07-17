from datetime import datetime, timedelta, timezone

from app.domain.nrim.alert_policy import (
    AlertPolicy,
    AlertState,
    decide_alert_state,
)
from app.domain.nrim.temporal import (
    EarlyWarning,
    PrecursorEvidence,
    WarningTier,
)
from app.domain.nrim.telemetry import TelemetryQuality


def make_warning(
    tier: WarningTier,
) -> EarlyWarning:
    return EarlyWarning(
        case_id="case_1",
        deployment_id="deployment_1",
        component_node_id="storage_1",
        failure_signature_id=(
            "FS-CATCHUP-STORAGE-PRECURSOR-001"
        ),
        tier=tier,
        title="Storage warning",
        conclusion="Test warning.",
        evidence=[
            PrecursorEvidence(
                evidence_id="evidence_1",
                signal_name="storage_usage",
                event_ids=["event_1"],
                statement="Storage usage is increasing.",
                strength=0.9,
                quality=TelemetryQuality.HIGH,
            )
        ],
        requires_engineer_review=(
            tier
            in {
                WarningTier.HIGH_RISK,
                WarningTier.ACTIVE_INCIDENT,
            }
        ),
    )


def test_first_early_warning_is_pending() -> None:
    decision = decide_alert_state(
        warning=make_warning(
            WarningTier.EARLY_WARNING
        ),
        observed_at=datetime.now(timezone.utc),
        previous_decision=None,
        policy=AlertPolicy(
            minimum_consecutive_windows=2
        ),
    )

    assert decision.state == AlertState.PENDING


def test_second_early_warning_opens_alert() -> None:
    now = datetime.now(timezone.utc)

    first = decide_alert_state(
        warning=make_warning(
            WarningTier.EARLY_WARNING
        ),
        observed_at=now,
        previous_decision=None,
    )

    second = decide_alert_state(
        warning=make_warning(
            WarningTier.EARLY_WARNING
        ),
        observed_at=now + timedelta(minutes=5),
        previous_decision=first,
    )

    assert second.state == AlertState.OPEN


def test_high_risk_opens_immediately() -> None:
    warning = make_warning(WarningTier.HIGH_RISK)

    decision = decide_alert_state(
        warning=warning,
        observed_at=datetime.now(timezone.utc),
        previous_decision=None,
    )

    assert decision.state == AlertState.OPEN
