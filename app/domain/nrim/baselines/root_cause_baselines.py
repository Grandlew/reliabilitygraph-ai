from __future__ import annotations

import hashlib
from typing import Any

from .feature_access import (
    FeatureAccessor,
    node_rows_by_id,
)
from .models import (
    BaselineName,
    NodeScore,
)

from .topology_scoring import (
    topology_propagation_baseline,
)

NODE_TYPE_CRITICALITY = {
    "catchup_storage": 1.00,
    "catchup_service": 0.95,
    "middleware": 0.90,
    "database": 0.85,
    "core_switch": 0.80,
    "streamer": 0.75,
    "gateway": 0.70,
    "distribution_switch": 0.60,
    "epg_service": 0.55,
    "signal_source": 0.50,
    "smart_tv_group": 0.30,
}


ANOMALY_FEATURE_WEIGHTS = {
    "system__disk__utilization__latest": 0.20,
    "system__disk__utilization__maximum": 0.15,
    "system__disk__utilization__slope": 0.10,
    "system__disk__io_latency__latest": 0.20,
    "system__disk__io_latency__maximum": 0.10,
    "system__disk__io_errors__maximum": 0.10,
    "iptv__catchup__recording_failures__maximum": 0.10,
    "system__process__restart_count__maximum": 0.05,
}


ERROR_FEATURE_WEIGHTS = {
    "system__disk__io_errors__latest": 0.35,
    "system__disk__io_errors__maximum": 0.25,
    "iptv__catchup__recording_failures__latest": 0.20,
    "iptv__catchup__recording_failures__maximum": 0.15,
    "system__process__restart_count__maximum": 0.05,
}

# Scale mapping for robust_scale (x / (x + scale))
FEATURE_SCALES = {
    "utilization": 0.5,
    "latency": 100.0,
    "errors": 5.0,
    "count": 1.0,
    "slope": 0.1,
    "default": 10.0,
}


def deterministic_random_score(
    *,
    window_id: str,
    node_id: str,
    seed: int,
) -> float:
    payload = (
        f"{seed}:{window_id}:{node_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).hexdigest()

    integer = int(digest[:16], 16)

    return integer / float(16**16 - 1)


def random_baseline(
    *,
    window: dict[str, Any],
    seed: int = 42,
) -> list[NodeScore]:
    return [
        NodeScore(
            node_id=str(node_id),
            score=deterministic_random_score(
                window_id=str(window["window_id"]),
                node_id=str(node_id),
                seed=seed,
            ),
            evidence={"random": 1.0},
        )
        for node_id in window["node_ids"]
    ]


def infer_node_type(
    *,
    row: list[float],
    accessor: FeatureAccessor,
) -> str | None:
    candidates = []

    for node_type in NODE_TYPE_CRITICALITY:
        feature_name = (
            f"node_type__{node_type}"
        )

        value = accessor.get(
            row,
            feature_name,
        )

        if value > 0.5:
            candidates.append(node_type)

    if len(candidates) > 1:
        raise ValueError(
            "Node has multiple active node types."
        )

    return candidates[0] if candidates else None


def static_criticality_baseline(
    *,
    window: dict[str, Any],
) -> list[NodeScore]:
    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )
    rows = node_rows_by_id(window)

    scores = []

    for node_id, row in rows.items():
        node_type = infer_node_type(
            row=row,
            accessor=accessor,
        )

        score = NODE_TYPE_CRITICALITY.get(
            node_type or "",
            0.0,
        )

        scores.append(
            NodeScore(
                node_id=node_id,
                score=score,
                evidence={
                    "static_criticality": score
                },
            )
        )

    return scores


def robust_scale(
    value: float,
    *,
    scale: float,
) -> float:
    if scale <= 0:
        raise ValueError(
            "scale must be positive."
        )

    non_negative = max(0.0, value)

    return non_negative / (
        non_negative + scale
    )


def weighted_feature_score(
    *,
    row: list[float],
    accessor: FeatureAccessor,
    feature_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    total_score = 0.0
    evidence = {}

    for feature_name, weight in feature_weights.items():
        # Access value safely
        raw_value = accessor.get(row, feature_name, default=0.0)

        # Determine scale based on feature name
        scale = FEATURE_SCALES["default"]
        for key, val in FEATURE_SCALES.items():
            if key in feature_name:
                scale = val
                break

        # Normalize and weight
        contribution = robust_scale(raw_value, scale=scale)
        weighted_val = contribution * weight

        total_score += weighted_val
        evidence[feature_name] = weighted_val

    return total_score, evidence


def anomaly_baseline(
    *,
    window: dict[str, Any],
) -> list[NodeScore]:
    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )

    rows = node_rows_by_id(window)
    scores = []

    for node_id, row in rows.items():
        score, evidence = weighted_feature_score(
            row=row,
            accessor=accessor,
            feature_weights=(
                ANOMALY_FEATURE_WEIGHTS
            ),
        )

        scores.append(
            NodeScore(
                node_id=node_id,
                score=score,
                evidence=evidence,
            )
        )

    return scores


def error_evidence_baseline(
    *,
    window: dict[str, Any],
) -> list[NodeScore]:
    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )
    rows = node_rows_by_id(window)

    scores = []

    for node_id, row in rows.items():
        score, evidence = weighted_feature_score(
            row=row,
            accessor=accessor,
            feature_weights=(
                ERROR_FEATURE_WEIGHTS
            ),
        )

        scores.append(
            NodeScore(
                node_id=node_id,
                score=score,
                evidence=evidence,
            )
        )

    return scores


def score_map(
    scores: list[NodeScore],
) -> dict[str, NodeScore]:
    return {
        item.node_id: item
        for item in scores
    }


def hybrid_engineering_baseline(
    *,
    window: dict[str, Any],
    anomaly_weight: float = 0.35,
    error_weight: float = 0.25,
    topology_weight: float = 0.25,
    criticality_weight: float = 0.15,
    missingness_penalty: float = 0.10,
) -> list[NodeScore]:
    total_weight = (
        anomaly_weight
        + error_weight
        + topology_weight
        + criticality_weight
    )

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            "Positive hybrid weights must sum to 1."
        )

    anomaly = anomaly_baseline(window=window)
    errors = error_evidence_baseline(window=window)
    criticality = static_criticality_baseline(
        window=window
    )

    topology = topology_propagation_baseline(
        window=window,
        local_scores=anomaly,
    )

    anomaly_by_id = score_map(anomaly)
    errors_by_id = score_map(errors)
    criticality_by_id = score_map(criticality)
    topology_by_id = score_map(topology)

    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )
    rows = node_rows_by_id(window)

    results = []

    for node_id in window["node_ids"]:
        node_id = str(node_id)

        row = rows[node_id]

        missing_features = [
            name
            for name in accessor.feature_names
            if name.endswith("__missing")
        ]

        missing_fraction = (
            sum(
                accessor.get(row, name)
                for name in missing_features
            )
            / len(missing_features)
            if missing_features
            else 0.0
        )

        components = {
            "anomaly": (
                anomaly_by_id[node_id].score
            ),
            "error": errors_by_id[node_id].score,
            "topology": (
                topology_by_id[node_id].score
            ),
            "criticality": (
                criticality_by_id[node_id].score
            ),
        }

        positive_score = (
            anomaly_weight * components["anomaly"]
            + error_weight * components["error"]
            + topology_weight * components["topology"]
            + criticality_weight
            * components["criticality"]
        )

        penalty = (
            missingness_penalty
            * missing_fraction
        )

        final_score = max(
            0.0,
            positive_score - penalty,
        )

        results.append(
            NodeScore(
                node_id=node_id,
                score=final_score,
                evidence={
                    **components,
                    "missing_fraction": (
                        missing_fraction
                    ),
                    "missingness_penalty": penalty,
                },
            )
        )

    return results


def rank_node_scores(
    scores: list[NodeScore],
) -> list[NodeScore]:

    node_ids = [s.node_id for s in scores]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node IDs found in scores list")

    # Sort by score descending, then node_id ascending for ties
    return sorted(
        scores,
        key=lambda x: (-x.score, x.node_id)
    )


def run_baseline(
    *,
    window: dict[str, Any],
    baseline: BaselineName,
    random_seed: int = 42,
) -> list[NodeScore]:
    if baseline == BaselineName.RANDOM:
        return random_baseline(
            window=window,
            seed=random_seed,
        )

    if baseline == BaselineName.STATIC_CRITICALITY:
        return static_criticality_baseline(
            window=window
        )

    if baseline == BaselineName.MAX_ANOMALY:
        return anomaly_baseline(window=window)

    if baseline == BaselineName.ERROR_EVIDENCE:
        return error_evidence_baseline(
            window=window
        )

    if baseline == BaselineName.TOPOLOGY_PROPAGATION:
        return topology_propagation_baseline(
            window=window,
            local_scores=anomaly_baseline(
                window=window
            ),
        )

    if baseline == BaselineName.HYBRID_ENGINEERING:
        return hybrid_engineering_baseline(
            window=window
        )

    raise ValueError(
        f"Unsupported baseline: {baseline.value}"
    )
