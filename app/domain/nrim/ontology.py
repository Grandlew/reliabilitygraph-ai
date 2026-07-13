from __future__ import annotations

from enum import Enum


class ProductMode(str, Enum):
    ARCHITECTURE_ASSURANCE = "architecture_assurance"
    OPERATIONAL_HEALTH = "operational_health"
    ROOT_CAUSE_DIAGNOSIS = "root_cause_diagnosis"
    PREDICTIVE_RELIABILITY = "predictive_reliability"


class NodeCategory(str, Enum):
    REQUIREMENT = "requirement"
    ARCHITECTURE = "architecture"
    FAILURE = "failure"
    REASONING = "reasoning"
    ACTION = "action"
    OBSERVATION = "observation"
    CHANGE = "change"


class KnowledgeStatus(str, Enum):
    CONFIRMED = "confirmed"
    EXTRACTED = "extracted"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    RECOMMENDED = "recommended"
    OBSERVED = "observed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class EvidenceKind(str, Enum):
    USER_STATEMENT = "user_statement"
    STRUCTURED_FORM = "structured_form"
    DOCUMENT_EXTRACT = "document_extract"
    PRODUCT_DOCUMENTATION = "product_documentation"
    DETERMINISTIC_RULE = "deterministic_rule"
    CONFIGURATION_EXPORT = "configuration_export"
    METRIC = "metric"
    LOG = "log"
    ALARM = "alarm"
    DIAGNOSTIC_TEST = "diagnostic_test"
    ENGINEER_CONFIRMATION = "engineer_confirmation"
    HISTORICAL_CASE = "historical_case"
    SYSTEM_ASSUMPTION = "system_assumption"


class SignalType(str, Enum):
    METRIC = "metric"
    LOG = "log"
    EVENT = "event"
    ALARM = "alarm"
    CONFIGURATION_CHANGE = "configuration_change"
    MAINTENANCE_ACTION = "maintenance_action"
    EXTERNAL_PROBE = "external_probe"
    USER_REPORTED_SYMPTOM = "user_reported_symptom"


class GoldenSignal(str, Enum):
    LATENCY = "latency"
    TRAFFIC = "traffic"
    ERRORS = "errors"
    SATURATION = "saturation"


class OutcomeStatus(str, Enum):
    SOLVED = "solved"
    PARTIALLY_SOLVED = "partially_solved"
    NOT_SOLVED = "not_solved"
    UNKNOWN = "unknown"


class FailurePredictability(str, Enum):
    TREND_PREDICTABLE = "trend_predictable"
    CONDITION_PREDICTABLE = "condition_predictable"
    CHANGE_INDUCED = "change_induced"
    ABRUPT = "abrupt"


class EdgeType(str, Enum):
    REQUIRES = "requires"
    CONSTRAINED_BY = "constrained_by"
    CONFLICTS_WITH = "conflicts_with"
    SATISFIED_BY = "satisfied_by"
    PARTIALLY_SATISFIED_BY = "partially_satisfied_by"
    NOT_SATISFIED_BY = "not_satisfied_by"

    DEPENDS_ON = "depends_on"
    RECEIVES_FROM = "receives_from"
    SENDS_TO = "sends_to"
    CONTROLS = "controls"
    AUTHENTICATES_WITH = "authenticates_with"
    STORES_FOR = "stores_for"
    INTEGRATES_WITH = "integrates_with"
    SERVES = "serves"
    BELONGS_TO_VLAN = "belongs_to_vlan"
    MONITORED_BY = "monitored_by"

    CAN_CAUSE = "can_cause"
    CONTRIBUTES_TO = "contributes_to"
    PROPAGATES_TO = "propagates_to"
    MANIFESTS_AS = "manifests_as"
    INCREASES_RISK_OF = "increases_risk_of"
    PREVENTED_BY = "prevented_by"
    DETECTED_BY = "detected_by"

    SUPPORTED_BY = "supported_by"
    CONTRADICTED_BY = "contradicted_by"
    DERIVED_FROM = "derived_from"
    CONFIRMED_BY = "confirmed_by"
    REJECTED_BY = "rejected_by"
    SIMILAR_TO_CASE = "similar_to_case"

    TESTS_HYPOTHESIS = "tests_hypothesis"
    DISTINGUISHES_BETWEEN = "distinguishes_between"
    RULES_OUT = "rules_out"

    ADDRESSES = "addresses"
    EXECUTED_ON = "executed_on"
    PRODUCED_OUTCOME = "produced_outcome"
    VERIFIED_BY = "verified_by"
    RESOLVED_BY = "resolved_by"
    LEARNED_FROM = "learned_from"

    OBSERVED_ON = "observed_on"
    PRECEDED = "preceded"
    DEVIATES_FROM_BASELINE = "deviates_from_baseline"
    PREDICTED_TO_CAUSE = "predicted_to_cause"
