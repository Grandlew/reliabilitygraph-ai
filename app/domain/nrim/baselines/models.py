from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BaselineName(str, Enum):
    RANDOM = "random"
    STATIC_CRITICALITY = "static_criticality"
    MAX_ANOMALY = "max_anomaly"
    ERROR_EVIDENCE = "error_evidence"
    TOPOLOGY_PROPAGATION = "topology_propagation"
    HYBRID_ENGINEERING = "hybrid_engineering"
    LEARNED_FUSION = "learned_fusion"


class NodeScore(BaseModel):
    node_id: str = Field(min_length=1)
    score: float
    evidence: dict[str, float] = Field(
        default_factory=dict
    )


class WindowRankingResult(BaseModel):
    window_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    baseline: BaselineName

    node_scores: list[NodeScore] = Field(
        min_length=1
    )
    ranked_node_ids: list[str] = Field(
        min_length=1
    )

    true_root_cause_node_id: str | None
    true_root_cause_rank: int | None = Field(
        default=None,
        ge=1,
    )

    true_incident: bool | None = None
    incident_detected: bool | None = None
    incident_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    ood_score: float | None = Field(default=None, ge=0.0)
    ood_detected: bool = False
    incident_escalated: bool = False

    abstained: bool = False
    top_score: float
    score_margin: float = Field(ge=0.0)

    runtime_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ranking(
        self,
    ) -> "WindowRankingResult":
        scored_node_ids = {n.node_id for n in self.node_scores}
        ranked_node_id_set = set(self.ranked_node_ids)

        if len(self.ranked_node_ids) != len(ranked_node_id_set):
            raise ValueError("Duplicate node IDs found in ranked_node_ids")

        if scored_node_ids != ranked_node_id_set:
            raise ValueError(
                "ranked_node_ids must contain exactly the same IDs as node_scores")
        if self.true_root_cause_node_id and not self.abstained and self.true_root_cause_rank is not None:
            try:
                actual_index = self.ranked_node_ids.index(
                    self.true_root_cause_node_id)
                if actual_index + 1 != self.true_root_cause_rank:
                    raise ValueError(
                        f"true_root_cause_rank ({self.true_root_cause_rank}) does not match "
                        f"position in ranked_node_ids (index {actual_index} + 1)"
                    )
            except ValueError:
                raise ValueError(
                    f"True root cause {self.true_root_cause_node_id} not found in ranked_node_ids")

        return self


class RankingMetricSummary(BaseModel):
    window_count: int = Field(ge=0)
    evaluated_faulty_windows: int = Field(ge=0)

    mrr: float = Field(ge=0.0, le=1.0)
    hits_at_1: float = Field(ge=0.0, le=1.0)
    hits_at_3: float = Field(ge=0.0, le=1.0)

    mean_rank: float | None = Field(
        default=None,
        ge=1.0,
    )
    median_rank: float | None = Field(
        default=None,
        ge=1.0,
    )

    healthy_window_count: int = Field(ge=0)
    healthy_abstention_rate: float = Field(
        ge=0.0,
        le=1.0,
    )
    healthy_false_selection_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    faulty_coverage: float = Field(
        ge=0.0,
        le=1.0,
    )
    selective_mrr: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_runtime_ms: float = Field(ge=0.0)
    incident_precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    incident_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    ood_detection_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    incident_escalation_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class BaselineEvaluationResult(BaseModel):
    baseline: BaselineName
    benchmark_fingerprint: str = Field(
        min_length=64,
        max_length=64,
    )

    abstention_threshold: float | None = None
    incident_threshold: float | None = None
    incident_threshold_feasible: bool = False

    validation: RankingMetricSummary
    test: RankingMetricSummary
    ood_test: RankingMetricSummary

    configuration: dict[str, Any]
