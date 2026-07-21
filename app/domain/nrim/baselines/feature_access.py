from __future__ import annotations

from typing import Any


class FeatureAccessor:
    def __init__(
        self,
        feature_names: list[str],
    ) -> None:
        if len(feature_names) != len(
            set(feature_names)
        ):
            raise ValueError(
                "Feature names must be unique."
            )

        self.feature_names = list(feature_names)
        self.index_by_name = {
            name: index
            for index, name in enumerate(
                self.feature_names
            )
        }

    def has(self, name: str) -> bool:
        return name in self.index_by_name

    def get(
        self,
        row: list[float],
        name: str,
        *,
        default: float = 0.0,
    ) -> float:
        index = self.index_by_name.get(name)

        if index is None:
            return default

        if index >= len(row):
            raise ValueError(
                f"Feature row is too short for '{name}'."
            )

        return float(row[index])

    def get_required(
        self,
        row: list[float],
        name: str,
    ) -> float:
        if name not in self.index_by_name:
            raise KeyError(
                f"Required feature not found: {name}"
            )

        return self.get(row, name)


def node_rows_by_id(
    window: dict[str, Any],
) -> dict[str, list[float]]:
    node_ids = list(window["node_ids"])
    matrix = list(window["node_features"])

    if len(node_ids) != len(matrix):
        raise ValueError(
            "Node IDs and node-feature rows differ."
        )

    return {
        str(node_id): [
            float(value)
            for value in row
        ]
        for node_id, row in zip(
            node_ids,
            matrix,
            strict=True,
        )
    }


def true_root_cause_node_id(
    window: dict[str, Any],
) -> str | None:
    node_ids = list(window["node_ids"])
    labels = list(
        window["targets"]["root_cause_node"]
    )

    if len(node_ids) != len(labels):
        raise ValueError(
            "Root-cause target length differs "
            "from node count."
        )

    positives = [
        str(node_id)
        for node_id, label in zip(
            node_ids,
            labels,
            strict=True,
        )
        if int(label) == 1
    ]
    if not positives:
        return None
    if len(positives) == 1:
        return positives[0]
    if len(positives) > 1:
        raise ValueError("Multiple root cause nodes found")
