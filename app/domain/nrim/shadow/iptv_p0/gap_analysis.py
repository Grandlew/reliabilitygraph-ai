from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..hashing import canonical_hash


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class CoverageArea(str, Enum):
    SOURCE = "source"
    SCHEMA = "schema"
    CLOCK = "clock"
    TOPOLOGY = "topology"
    FEATURE = "feature"
    OUTCOME = "outcome"


class SafeRoute(str, Enum):
    UNKNOWN = "UNKNOWN"
    ESCALATE = "ESCALATE"
    BLOCKED = "BLOCKED"


_SEVERITY = {
    SafeRoute.UNKNOWN: 0,
    SafeRoute.ESCALATE: 1,
    SafeRoute.BLOCKED: 2,
}


class CoverageMeasurement(StrictModel):
    area: CoverageArea
    subject_id: str = Field(min_length=1, max_length=256)
    expected_count: int = Field(ge=0)
    qualified_count: int = Field(ge=0)
    minimum_fraction: float = Field(ge=0.0, le=1.0)
    failure_route: SafeRoute

    @property
    def fraction(self) -> float:
        if self.expected_count == 0:
            return 1.0
        return self.qualified_count / self.expected_count


class Gap(StrictModel):
    area: CoverageArea
    subject_id: str
    reason_code: str
    route: SafeRoute
    measured_fraction: float
    required_fraction: float


class GapReport(StrictModel):
    measurements: tuple[CoverageMeasurement, ...]
    gaps: tuple[Gap, ...]
    terminal_route: SafeRoute | None

    @property
    def passed(self) -> bool:
        return not self.gaps

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


def compile_gap_report(
    measurements: tuple[CoverageMeasurement, ...],
) -> GapReport:
    if not measurements:
        raise ValueError("Gap analysis requires coverage measurements")
    keys = [(item.area, item.subject_id) for item in measurements]
    if len(keys) != len(set(keys)):
        raise ValueError("Coverage measurements must be unique")
    gaps = tuple(
        Gap(
            area=item.area,
            subject_id=item.subject_id,
            reason_code=f"{item.area.value.upper()}_COVERAGE_BELOW_MINIMUM",
            route=item.failure_route,
            measured_fraction=item.fraction,
            required_fraction=item.minimum_fraction,
        )
        for item in sorted(
            measurements,
            key=lambda value: (value.area.value, value.subject_id),
        )
        if item.fraction < item.minimum_fraction
    )
    terminal = (
        max((item.route for item in gaps), key=_SEVERITY.get)
        if gaps
        else None
    )
    return GapReport(
        measurements=tuple(
            sorted(
                measurements,
                key=lambda value: (value.area.value, value.subject_id),
            )
        ),
        gaps=gaps,
        terminal_route=terminal,
    )
