from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RuleEffect(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    REQUIRE = "require"


@dataclass(frozen=True)
class EvidenceRule:
    rule_id: str
    hypothesis_type: str
    signal_name: str
    comparison: str
    threshold: float | None
    effect: RuleEffect
    strength: str
    explanation: str
    required_attributes: dict[str, str] = field(default_factory=dict)


CATCHUP_STORAGE_RULES: tuple[EvidenceRule, ...] = (
    EvidenceRule(
        rule_id="RULE_STORAGE_UTILIZATION_HIGH",
        hypothesis_type="storage_capacity_saturation",
        signal_name="system.disk.utilization",
        comparison="greater_than_or_equal",
        threshold=85.0,
        effect=RuleEffect.SUPPORT,
        strength="moderate",
        explanation=(
            "High storage utilization supports a possible "
            "storage-capacity saturation hypothesis."
        ),
    ),
    EvidenceRule(
        rule_id="RULE_STORAGE_UTILIZATION_CRITICAL",
        hypothesis_type="storage_capacity_saturation",
        signal_name="system.disk.utilization",
        comparison="greater_than_or_equal",
        threshold=95.0,
        effect=RuleEffect.SUPPORT,
        strength="strong",
        explanation=(
            "Critical storage utilization strongly supports "
            "storage-capacity saturation."
        ),
    ),
    EvidenceRule(
        rule_id="RULE_STORAGE_LATENCY_HIGH",
        hypothesis_type="storage_io_degradation",
        signal_name="system.disk.io_latency",
        comparison="greater_than_or_equal",
        threshold=30.0,
        effect=RuleEffect.SUPPORT,
        strength="moderate",
        explanation=(
            "Elevated storage I/O latency supports storage-path "
            "degradation."
        ),
    ),
    EvidenceRule(
        rule_id="RULE_CATCHUP_RECORDING_FAILURES",
        hypothesis_type="storage_io_degradation",
        signal_name="iptv.catchup.recording_failures",
        comparison="greater_than",
        threshold=0.0,
        effect=RuleEffect.SUPPORT,
        strength="moderate",
        explanation=(
            "CatchUP recording failures are consistent with "
            "storage-path degradation."
        ),
    ),
    EvidenceRule(
        rule_id="RULE_STORAGE_UTILIZATION_NORMAL",
        hypothesis_type="storage_capacity_saturation",
        signal_name="system.disk.utilization",
        comparison="less_than",
        threshold=70.0,
        effect=RuleEffect.CONTRADICT,
        strength="moderate",
        explanation=(
            "Normal storage utilization weakens the capacity "
            "saturation hypothesis."
        ),
    ),
)
