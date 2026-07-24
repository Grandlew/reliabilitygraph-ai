from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .contracts import (
    Applicability,
    ComponentType,
    MetricName,
    MetricUnit,
    ObservationQuality,
    TelemetryObservation,
)
from .hashing import canonical_hash
from .privacy import (
    Pseudonymizer,
    PrivacyViolation,
    assert_no_direct_identifiers,
)


class AggregationStatistic(str, Enum):
    LAST = "last"
    MEAN = "mean"
    MAXIMUM = "maximum"
    COUNT = "count"
    RATE = "rate"


class ResetBehavior(str, Enum):
    GAUGE = "gauge"
    INTERVAL_COUNTER = "interval_counter"
    MONOTONIC_COUNTER = "monotonic_counter"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class AggregationSemantics(StrictModel):
    statistic: AggregationStatistic
    source_sampling_interval_seconds: float = Field(gt=0.0)
    target_window_seconds: float = Field(gt=0.0)
    reset_behavior: ResetBehavior
    numerator_semantics: str = Field(min_length=3, max_length=500)
    denominator_semantics: str = Field(min_length=3, max_length=500)
    boundary_rule: str = Field(min_length=3, max_length=500)
    late_data_rule: str = Field(min_length=3, max_length=500)


class MetricSemanticMapping(StrictModel):
    source_metric_name: str = Field(min_length=1, max_length=256)
    canonical_metric: MetricName
    expected_source_unit: str = Field(min_length=1, max_length=128)
    canonical_unit: MetricUnit
    conversion_multiplier: float
    conversion_offset: float = 0.0
    aggregation: AggregationSemantics
    conversion_provenance: str = Field(min_length=10, max_length=1000)

    @field_validator("conversion_multiplier", "conversion_offset")
    @classmethod
    def finite_conversion(cls, value: float) -> float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("Unit conversions must be finite")
        return value


class CollectorSemanticContract(StrictModel):
    contract_id: str = Field(min_length=1, max_length=128)
    contract_version: str = Field(min_length=1, max_length=128)
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    collector_id: str = Field(min_length=1, max_length=128)
    metric_field: str = Field(min_length=1, max_length=128)
    value_field: str = Field(min_length=1, max_length=128)
    unit_field: str = Field(min_length=1, max_length=128)
    event_time_field: str = Field(min_length=1, max_length=128)
    ingestion_time_field: str = Field(min_length=1, max_length=128)
    component_field: str = Field(min_length=1, max_length=128)
    sequence_field: str = Field(min_length=1, max_length=128)
    topology_version_field: str = Field(min_length=1, max_length=128)
    quality_field: str = Field(min_length=1, max_length=128)
    applicability_field: str = Field(min_length=1, max_length=128)
    event_timestamp_provenance: str = Field(min_length=10, max_length=1000)
    clock_domain: str = Field(min_length=3, max_length=256)
    timestamp_precision_ms: float = Field(gt=0.0)
    maximum_clock_skew_seconds: float = Field(ge=0.0)
    maximum_ingestion_delay_seconds: float = Field(gt=0.0)
    quality_mapping: dict[str, ObservationQuality] = Field(min_length=3)
    applicability_mapping: dict[str, Applicability] = Field(min_length=3)
    metric_mappings: tuple[MetricSemanticMapping, ...]
    component_types: dict[str, ComponentType] = Field(min_length=1)
    approved_mapping_evidence: str = Field(min_length=10, max_length=1000)

    @model_validator(mode="after")
    def validate_completeness(self):
        canonical = [item.canonical_metric for item in self.metric_mappings]
        if set(canonical) != set(MetricName) or len(canonical) != len(
            set(canonical)
        ):
            raise ValueError(
                "Collector contract must map every frozen metric exactly once"
            )
        source = [item.source_metric_name for item in self.metric_mappings]
        if len(source) != len(set(source)):
            raise ValueError("Source metric names must be unique")
        if set(self.quality_mapping.values()) != set(ObservationQuality):
            raise ValueError(
                "Quality mapping must represent all frozen quality states"
            )
        if set(self.applicability_mapping.values()) != set(Applicability):
            raise ValueError(
                "Applicability mapping must distinguish observed, missing "
                "and not-applicable"
            )
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class SemanticFixture(StrictModel):
    fixture_id: str = Field(min_length=1, max_length=128)
    raw_record: dict[str, Any]
    expected_metric: MetricName
    expected_value: float | None
    expected_quality: ObservationQuality
    expected_applicability: Applicability


class SemanticAcceptanceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class SemanticAcceptanceReport:
    contract_hash: str
    required_mapping_fraction: float
    fixture_match_fraction: float
    mutation_rejection_fraction: float
    fixture_results: tuple[dict[str, Any], ...]
    mutation_results: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        return (
            self.required_mapping_fraction == 1.0
            and self.fixture_match_fraction == 1.0
            and self.mutation_rejection_fraction == 1.0
        )


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise SemanticAcceptanceError(
            "TIMESTAMP_TYPE",
            f"{field} must be an ISO-8601 string",
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SemanticAcceptanceError(
            "TIMESTAMP_PARSE",
            f"{field} is not ISO-8601",
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SemanticAcceptanceError(
            "TIMESTAMP_TIMEZONE",
            f"{field} must include an offset",
        )
    return parsed.astimezone(timezone.utc)


class CollectorAcceptanceHarness:
    def __init__(
        self,
        *,
        contract: CollectorSemanticContract,
        pseudonymizer: Pseudonymizer,
    ) -> None:
        self.contract = contract
        self.pseudonymizer = pseudonymizer
        self._mappings = {
            item.source_metric_name: item
            for item in contract.metric_mappings
        }

    @staticmethod
    def _required(record: Mapping[str, Any], field: str) -> Any:
        if field not in record:
            raise SemanticAcceptanceError(
                "MISSING_SOURCE_FIELD",
                f"Required source field is absent: {field}",
            )
        return record[field]

    def map_record(self, record: Mapping[str, Any]) -> TelemetryObservation:
        try:
            assert_no_direct_identifiers(record)
        except PrivacyViolation as error:
            raise SemanticAcceptanceError(
                "DIRECT_IDENTIFIER",
                str(error),
            ) from error
        metric_source = str(
            self._required(record, self.contract.metric_field)
        )
        mapping = self._mappings.get(metric_source)
        if mapping is None:
            raise SemanticAcceptanceError(
                "UNKNOWN_METRIC",
                metric_source,
            )
        unit = str(self._required(record, self.contract.unit_field))
        if unit != mapping.expected_source_unit:
            raise SemanticAcceptanceError(
                "UNIT_MISMATCH",
                f"{unit!r} != {mapping.expected_source_unit!r}",
            )
        event_time = _timestamp(
            self._required(record, self.contract.event_time_field),
            self.contract.event_time_field,
        )
        ingestion_time = _timestamp(
            self._required(record, self.contract.ingestion_time_field),
            self.contract.ingestion_time_field,
        )
        future_skew = (event_time - ingestion_time).total_seconds()
        if future_skew > self.contract.maximum_clock_skew_seconds:
            raise SemanticAcceptanceError(
                "CLOCK_SKEW",
                "Event time exceeds registered clock-skew budget",
            )
        ingestion_delay = (ingestion_time - event_time).total_seconds()
        if (
            ingestion_delay
            > self.contract.maximum_ingestion_delay_seconds
        ):
            raise SemanticAcceptanceError(
                "INGESTION_DELAY",
                "Observation exceeds registered ingestion-delay budget",
            )
        raw_quality = str(
            self._required(record, self.contract.quality_field)
        )
        quality = self.contract.quality_mapping.get(raw_quality)
        if quality is None:
            raise SemanticAcceptanceError(
                "UNKNOWN_QUALITY",
                raw_quality,
            )
        raw_applicability = str(
            self._required(record, self.contract.applicability_field)
        )
        applicability = self.contract.applicability_mapping.get(
            raw_applicability
        )
        if applicability is None:
            raise SemanticAcceptanceError(
                "UNKNOWN_APPLICABILITY",
                raw_applicability,
            )
        component = str(
            self._required(record, self.contract.component_field)
        )
        component_type = self.contract.component_types.get(component)
        if component_type is None:
            raise SemanticAcceptanceError(
                "UNKNOWN_COMPONENT",
                component,
            )
        raw_value = self._required(record, self.contract.value_field)
        value = None
        if applicability is Applicability.OBSERVED:
            if isinstance(raw_value, bool) or not isinstance(
                raw_value,
                (int, float),
            ):
                raise SemanticAcceptanceError(
                    "VALUE_TYPE",
                    "Observed metric value must be numeric",
                )
            value = (
                float(raw_value) * mapping.conversion_multiplier
                + mapping.conversion_offset
            )
        elif raw_value is not None:
            raise SemanticAcceptanceError(
                "NONOBSERVED_VALUE",
                "Missing/not-applicable source value must be null",
            )
        try:
            return TelemetryObservation(
                deployment_pseudonym=self.contract.deployment_pseudonym,
                component_pseudonym=self.pseudonymizer.pseudonymize(
                    component,
                    namespace="component",
                ),
                component_type=component_type,
                metric_name=mapping.canonical_metric,
                metric_value=value,
                unit=mapping.canonical_unit,
                applicability=applicability,
                quality=quality,
                event_time_utc=event_time,
                ingestion_time_utc=ingestion_time,
                collector_id=self.contract.collector_id,
                source_sequence_id=str(
                    self._required(record, self.contract.sequence_field)
                ),
                topology_version=str(
                    self._required(
                        record,
                        self.contract.topology_version_field,
                    )
                ),
            )
        except ValueError as error:
            raise SemanticAcceptanceError(
                "FROZEN_CONTRACT_MISMATCH",
                str(error),
            ) from error

    def check_fixture(self, fixture: SemanticFixture) -> dict[str, Any]:
        try:
            result = self.map_record(fixture.raw_record)
            matches = (
                result.metric_name is fixture.expected_metric
                and result.metric_value == fixture.expected_value
                and result.quality is fixture.expected_quality
                and result.applicability is fixture.expected_applicability
            )
            return {
                "fixture_id": fixture.fixture_id,
                "matches": matches,
                "error_code": None if matches else "SEMANTIC_MISMATCH",
                "observation_hash": canonical_hash(
                    result.model_dump(mode="json")
                ),
            }
        except SemanticAcceptanceError as error:
            return {
                "fixture_id": fixture.fixture_id,
                "matches": False,
                "error_code": error.code,
                "observation_hash": None,
            }

    def _negative_controls(
        self,
        fixture: SemanticFixture,
    ) -> tuple[dict[str, Any], ...]:
        baseline = dict(fixture.raw_record)
        unit_mutation = dict(baseline)
        unit_mutation[self.contract.unit_field] = "__wrong_unit__"
        timestamp_mutation = dict(baseline)
        ingestion = _timestamp(
            baseline[self.contract.ingestion_time_field],
            self.contract.ingestion_time_field,
        )
        timestamp_mutation[self.contract.event_time_field] = (
            ingestion
            + timedelta(
                seconds=self.contract.maximum_clock_skew_seconds + 1.0
            )
        ).isoformat()
        applicability_mutation = dict(baseline)
        applicability_mutation.pop(self.contract.applicability_field, None)
        quality_mutation = dict(baseline)
        quality_mutation[self.contract.quality_field] = "__unknown_quality__"
        controls: list[tuple[str, Mapping[str, Any], CollectorAcceptanceHarness]]
        controls = [
            ("swapped_unit", unit_mutation, self),
            ("shifted_timestamp", timestamp_mutation, self),
            ("removed_applicability", applicability_mutation, self),
            ("unknown_quality", quality_mutation, self),
        ]
        raw_quality = str(baseline[self.contract.quality_field])
        expected_quality = self.contract.quality_mapping[raw_quality]
        alternatives = [
            item for item in ObservationQuality if item is not expected_quality
        ]
        mutated_quality_mapping = dict(self.contract.quality_mapping)
        mutated_quality_mapping[raw_quality] = alternatives[0]
        semantic_mutation = CollectorAcceptanceHarness(
            contract=self.contract.model_copy(
                update={"quality_mapping": mutated_quality_mapping}
            ),
            pseudonymizer=self.pseudonymizer,
        )
        controls.append(
            ("mutated_quality_meaning", baseline, semantic_mutation)
        )
        results = []
        for name, record, harness in controls:
            outcome = harness.check_fixture(
                fixture.model_copy(update={"raw_record": dict(record)})
            )
            results.append(
                {
                    "mutation": name,
                    "rejected_or_mismatched": not outcome["matches"],
                    "error_code": outcome["error_code"],
                }
            )
        return tuple(results)

    def evaluate(
        self,
        fixtures: tuple[SemanticFixture, ...],
    ) -> SemanticAcceptanceReport:
        if not fixtures:
            raise ValueError("Semantic acceptance requires fixtures")
        fixture_results = tuple(
            self.check_fixture(item) for item in fixtures
        )
        mutation_results = tuple(
            result
            for fixture in fixtures
            for result in self._negative_controls(fixture)
        )
        mapped_metrics = {
            item.expected_metric for item in fixtures
        }
        return SemanticAcceptanceReport(
            contract_hash=self.contract.content_hash(),
            required_mapping_fraction=len(mapped_metrics) / len(MetricName),
            fixture_match_fraction=sum(
                bool(item["matches"]) for item in fixture_results
            )
            / len(fixture_results),
            mutation_rejection_fraction=sum(
                bool(item["rejected_or_mismatched"])
                for item in mutation_results
            )
            / len(mutation_results),
            fixture_results=fixture_results,
            mutation_results=mutation_results,
        )
