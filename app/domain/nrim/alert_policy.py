from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .temporal import EarlyWarning, WarningTier


class AlertState(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertDecision(BaseModel):
    alert_key: str
    state: AlertState
    tier: WarningTier
    reason: str

    first_seen_at: datetime
    last_seen_at: datetime
    consecutive_qualifying_windows: int = Field(ge=0)

    engineer_review_required: bool
    deduplication_key: str


class AlertPolicy(BaseModel):
    minimum_consecutive_windows: int = Field(default=2, ge=1)
    high_risk_immediate: bool = True
    active_incident_immediate: bool = True
    resolution_windows: int = Field(default=3, ge=1)
    cooldown_minutes: int = Field(default=30, ge=0)


def warning_deduplication_key(
    warning: EarlyWarning,
) -> str:
    return ":".join(
        [
            warning.deployment_id,
            warning.component_node_id,
            warning.failure_signature_id,
        ]
    )


def decide_alert_state(
    *,
    warning: EarlyWarning,
    observed_at: datetime,
    previous_decision: AlertDecision | None,
    policy: AlertPolicy | None = None,
) -> AlertDecision:
    policy = policy or AlertPolicy()
    deduplication_key = warning_deduplication_key(warning)

    immediate = (
        warning.tier == WarningTier.ACTIVE_INCIDENT
        and policy.active_incident_immediate
    ) or (
        warning.tier == WarningTier.HIGH_RISK
        and policy.high_risk_immediate
    )

    qualifies = warning.tier in {
        WarningTier.EARLY_WARNING,
        WarningTier.HIGH_RISK,
        WarningTier.ACTIVE_INCIDENT,
    }

    if previous_decision is None:
        consecutive = 1 if qualifies else 0
        first_seen = observed_at
    else:
        first_seen = previous_decision.first_seen_at
        consecutive = (
            previous_decision.consecutive_qualifying_windows + 1
            if qualifies
            else 0
        )

    if immediate:
        state = AlertState.OPEN
        reason = (
            "High-risk or active-incident policy allows "
            "immediate opening."
        )
    elif (
        qualifies
        and consecutive >= policy.minimum_consecutive_windows
    ):
        state = AlertState.OPEN
        reason = (
            "Warning persisted across the required number "
            "of consecutive windows."
        )
    elif qualifies:
        state = AlertState.PENDING
        reason = (
            "Warning detected but persistence requirement "
            "has not yet been met."
        )
    else:
        state = AlertState.SUPPRESSED
        reason = (
            "Current warning tier does not qualify for an alert."
        )

    return AlertDecision(
        alert_key=deduplication_key,
        state=state,
        tier=warning.tier,
        reason=reason,
        first_seen_at=first_seen,
        last_seen_at=observed_at,
        consecutive_qualifying_windows=consecutive,
        engineer_review_required=(
            warning.requires_engineer_review
        ),
        deduplication_key=deduplication_key,
    )
