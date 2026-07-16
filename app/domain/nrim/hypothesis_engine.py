from __future__ import annotations

from collections import defaultdict

from .reasoning import (
    EvidenceRole,
    EvidenceStrength,
    FailureHypothesis,
    HypothesisEvidenceLink,
    HypothesisStatus,
    ReasoningEvidence,
)
from .rule_catalog import EvidenceRule, RuleEffect
from .telemetry import CanonicalTelemetryEvent, TelemetryQuality

from .diagnostic_ranker import (
    create_catchup_diagnostic_options,
    rank_diagnostics,
)
from .intervention_engine import create_interventions
from .reasoning import ReasoningResult
from .rule_catalog import CATCHUP_STORAGE_RULES

STRENGTH_WEIGHTS = {
    EvidenceStrength.WEAK: 1.0,
    EvidenceStrength.MODERATE: 2.0,
    EvidenceStrength.STRONG: 4.0,
    EvidenceStrength.DEFINITIVE: 10.0,
}

QUALITY_MULTIPLIERS = {
    TelemetryQuality.VERIFIED: 1.00,
    TelemetryQuality.HIGH: 0.85,
    TelemetryQuality.MEDIUM: 0.60,
    TelemetryQuality.LOW: 0.30,
    TelemetryQuality.QUARANTINED: 0.00,
}


def compare_value(
    value: object,
    *,
    comparison: str,
    threshold: float | None,
) -> bool:
    if not isinstance(value, (int, float)):
        return False

    if threshold is None:
        return False

    numeric_value = float(value)

    if comparison == "greater_than":
        return numeric_value > threshold

    if comparison == "greater_than_or_equal":
        return numeric_value >= threshold

    if comparison == "less_than":
        return numeric_value < threshold

    if comparison == "less_than_or_equal":
        return numeric_value <= threshold

    if comparison == "equal":
        return numeric_value == threshold

    raise ValueError(f"Unsupported comparison: {comparison}")


def rule_matches(
    rule: EvidenceRule,
    event: CanonicalTelemetryEvent,
) -> bool:
    if event.signal_name != rule.signal_name:
        return False

    for attribute_key, expected_value in rule.required_attributes.items():
        actual_values = {
            attribute.key: str(attribute.value)
            for attribute in event.attributes
        }

        if actual_values.get(attribute_key) != expected_value:
            return False

    return compare_value(
        event.value,
        comparison=rule.comparison,
        threshold=rule.threshold,
    )


def evidence_role_from_rule(effect: RuleEffect) -> EvidenceRole:
    if effect == RuleEffect.SUPPORT:
        return EvidenceRole.SUPPORTS

    if effect == RuleEffect.CONTRADICT:
        return EvidenceRole.CONTRADICTS

    if effect == RuleEffect.REQUIRE:
        return EvidenceRole.REQUIRED

    return EvidenceRole.UNKNOWN


def strength_from_text(value: str) -> EvidenceStrength:
    try:
        return EvidenceStrength(value)
    except ValueError as error:
        raise ValueError(f"Unsupported evidence strength: {value}") from error


def calculate_link_weight(
    evidence: ReasoningEvidence,
) -> float:
    base_weight = STRENGTH_WEIGHTS[evidence.strength]
    quality_multiplier = QUALITY_MULTIPLIERS[evidence.quality]

    signed_weight = base_weight

    if evidence.role == EvidenceRole.CONTRADICTS:
        signed_weight *= -1.0
    elif evidence.role in {
        EvidenceRole.CONTEXT,
        EvidenceRole.UNKNOWN,
    }:
        signed_weight = 0.0

    return signed_weight * quality_multiplier


def build_evidence_links(
    *,
    hypothesis_id: str,
    hypothesis_type: str,
    events: list[CanonicalTelemetryEvent],
    rules: tuple[EvidenceRule, ...],
) -> list[HypothesisEvidenceLink]:
    links: list[HypothesisEvidenceLink] = []

    for event in events:
        for rule in rules:
            if rule.hypothesis_type != hypothesis_type:
                continue

            if not rule_matches(rule, event):
                continue

            evidence = ReasoningEvidence(
                evidence_id=f"{rule.rule_id}:{event.event_id}",
                source_event_id=event.event_id,
                statement=rule.explanation,
                role=evidence_role_from_rule(rule.effect),
                strength=strength_from_text(rule.strength),
                quality=event.quality,
                independent_source=event.collection_source,
                properties={
                    "signal_name": event.signal_name,
                    "observed_value": event.value,
                    "unit": event.unit,
                    "rule_id": rule.rule_id,
                },
            )

            links.append(
                HypothesisEvidenceLink(
                    hypothesis_id=hypothesis_id,
                    evidence=evidence,
                    weight=calculate_link_weight(evidence),
                )
            )

    return links


def calculate_evidence_coverage(
    links: list[HypothesisEvidenceLink],
) -> float:
    if not links:
        return 0.0

    independent_sources = {
        link.evidence.independent_source
        for link in links
        if link.evidence.quality != TelemetryQuality.QUARANTINED
    }

    represented_signal_names = {
        str(link.evidence.properties.get("signal_name"))
        for link in links
        if link.evidence.properties.get("signal_name")
    }

    support_present = any(
        link.evidence.role == EvidenceRole.SUPPORTS
        for link in links
    )

    contradiction_present = any(
        link.evidence.role == EvidenceRole.CONTRADICTS
        for link in links
    )

    source_score = min(len(independent_sources) / 3.0, 1.0)
    signal_score = min(len(represented_signal_names) / 3.0, 1.0)
    support_score = 1.0 if support_present else 0.0
    contradiction_awareness = 1.0 if contradiction_present else 0.5

    return round(
        (
            0.35 * source_score
            + 0.35 * signal_score
            + 0.20 * support_score
            + 0.10 * contradiction_awareness
        ),
        3,
    )


def score_hypothesis(
    hypothesis: FailureHypothesis,
) -> FailureHypothesis:
    score = sum(
        link.weight or 0.0
        for link in hypothesis.evidence_links
    )

    contradictions = [
        link.evidence.statement
        for link in hypothesis.evidence_links
        if link.evidence.role == EvidenceRole.CONTRADICTS
    ]

    support_count = sum(
        1
        for link in hypothesis.evidence_links
        if link.evidence.role == EvidenceRole.SUPPORTS
        and (link.weight or 0.0) > 0
    )

    status = HypothesisStatus.UNRESOLVED

    if support_count > 0 and score > 0:
        status = HypothesisStatus.SUPPORTED

    if score < 0:
        status = HypothesisStatus.WEAKENED

    return hypothesis.model_copy(
        update={
            "ranking_score": round(score, 3),
            "evidence_coverage": calculate_evidence_coverage(
                hypothesis.evidence_links
            ),
            "contradictions": contradictions,
            "status": status,
        }
    )


def mark_leading_hypothesis(
    hypotheses: list[FailureHypothesis],
) -> list[FailureHypothesis]:
    if not hypotheses:
        return []

    ranked = sorted(
        hypotheses,
        key=lambda item: (
            item.ranking_score,
            item.evidence_coverage,
        ),
        reverse=True,
    )

    top = ranked[0]

    if top.ranking_score <= 0:
        return ranked

    updated: list[FailureHypothesis] = []

    for hypothesis in ranked:
        if hypothesis.hypothesis_id == top.hypothesis_id:
            updated.append(
                hypothesis.model_copy(
                    update={
                        "status": HypothesisStatus.LEADING,
                    }
                )
            )
        else:
            updated.append(hypothesis)

    return updated


def create_catchup_storage_hypotheses() -> list[FailureHypothesis]:
    return [
        FailureHypothesis(
            hypothesis_id="hyp_storage_capacity",
            failure_type="storage_capacity_saturation",
            name="CatchUP storage capacity saturation",
            description=(
                "CatchUP recording failures may be caused by "
                "insufficient free storage capacity."
            ),
            target_node_ids=["arch_catchup_storage"],
            affected_service_node_ids=["service_catchup"],
            required_confirmation_evidence=[
                "Verify free capacity or allocation limit.",
                "Verify whether freeing capacity restores recording.",
            ],
            missing_evidence=[
                "Current free capacity trend.",
                "Cleanup-job status.",
                "Outcome after capacity remediation.",
            ],
        ),
        FailureHypothesis(
            hypothesis_id="hyp_storage_io",
            failure_type="storage_io_degradation",
            name="CatchUP storage I/O degradation",
            description=(
                "CatchUP recording failures may be caused by slow "
                "or unstable storage writes."
            ),
            target_node_ids=["arch_catchup_storage"],
            affected_service_node_ids=["service_catchup"],
            required_confirmation_evidence=[
                "Inspect write latency and I/O error evidence.",
                "Verify service improvement after storage remediation.",
            ],
            missing_evidence=[
                "Storage I/O queue depth.",
                "Filesystem or hardware errors.",
                "Per-channel recording distribution.",
            ],
        ),
        FailureHypothesis(
            hypothesis_id="hyp_catchup_application",
            failure_type="catchup_application_failure",
            name="CatchUP application or recording-worker failure",
            description=(
                "The service may be failing independently of "
                "storage capacity or storage performance."
            ),
            target_node_ids=["service_catchup"],
            affected_service_node_ids=["service_catchup"],
            required_confirmation_evidence=[
                "Inspect recording-worker logs.",
                "Verify worker health and restart behaviour.",
            ],
            missing_evidence=[
                "Application exception logs.",
                "Recording-worker state.",
                "Failure distribution across channels.",
            ],
        ),
    ]


def run_catchup_storage_reasoning(
    *,
    case_id: str,
    events: list[CanonicalTelemetryEvent],
) -> ReasoningResult:
    hypotheses = create_catchup_storage_hypotheses()

    scored_hypotheses: list[FailureHypothesis] = []

    for hypothesis in hypotheses:
        links = build_evidence_links(
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_type=hypothesis.failure_type,
            events=events,
            rules=CATCHUP_STORAGE_RULES,
        )

        hypothesis_with_links = hypothesis.model_copy(
            update={
                "evidence_links": links,
            }
        )

        scored_hypotheses.append(
            score_hypothesis(hypothesis_with_links)
        )

    ranked_hypotheses = mark_leading_hypothesis(
        scored_hypotheses
    )

    leading_ids = {
        hypothesis.hypothesis_id
        for hypothesis in ranked_hypotheses
        if hypothesis.status == HypothesisStatus.LEADING
    }

    diagnostics = rank_diagnostics(
        create_catchup_diagnostic_options(),
        leading_hypothesis_ids=leading_ids,
    )

    interventions = []

    for hypothesis in ranked_hypotheses:
        interventions.extend(
            create_interventions(hypothesis)
        )

    leading_hypothesis = next(
        (
            hypothesis
            for hypothesis in ranked_hypotheses
            if hypothesis.status == HypothesisStatus.LEADING
        ),
        None,
    )

    if leading_hypothesis is None:
        conclusion = (
            "No leading root-cause hypothesis can be selected from "
            "the available evidence."
        )
        leading_hypothesis_id = None
    else:
        conclusion = (
            f"The current leading hypothesis is "
            f"'{leading_hypothesis.name}'. This is not a confirmed "
            f"root cause. Additional diagnostic evidence is required."
        )
        leading_hypothesis_id = leading_hypothesis.hypothesis_id

    return ReasoningResult(
        case_id=case_id,
        hypotheses=ranked_hypotheses,
        ranked_diagnostics=diagnostics,
        recommended_interventions=interventions,
        leading_hypothesis_id=leading_hypothesis_id,
        conclusion=conclusion,
        limitations=[
            "The current telemetry is synthetic.",
            "Rule thresholds are provisional.",
            "No engineer-confirmed incident outcome exists.",
            "Ranking scores are not probabilities.",
        ],
    )
