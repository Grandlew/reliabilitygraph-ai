from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class FeatureValueType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    COUNT = "count"
    CATEGORICAL_ONE_HOT = "categorical_one_hot"


class FeatureDefinition(BaseModel):
    name: str = Field(min_length=1)
    value_type: FeatureValueType
    default_value: float
    requires_mask: bool = False
    description: str = Field(min_length=1)


NODE_TYPES = [
    "signal_source",
    "gateway",
    "streamer",
    "middleware",
    "database",
    "catchup_service",
    "catchup_storage",
    "epg_service",
    "core_switch",
    "distribution_switch",
    "smart_tv_group",
]


EDGE_TYPES = [
    "depends_on",
    "sends_to",
    "authenticates_with",
    "stores_on",
    "serves",
    "connected_to",
]


SIGNAL_NAMES = [
    "system.disk.utilization",
    "system.disk.io_latency",
    "system.disk.io_errors",
    "iptv.catchup.recording_failures",
    "system.process.restart_count",
    "iptv.session.active_count",
]


SIGNAL_STATISTICS = [
    "latest",
    "mean",
    "minimum",
    "maximum",
    "standard_deviation",
    "slope",
    "count",
    "low_quality_fraction",
    "hours_since_latest",
]


def build_node_feature_definitions() -> list[FeatureDefinition]:
    definitions: list[FeatureDefinition] = []

    for node_type in NODE_TYPES:
        definitions.append(
            FeatureDefinition(
                name=f"node_type__{node_type}",
                value_type=FeatureValueType.CATEGORICAL_ONE_HOT,
                default_value=0.0,
                description=f"One-hot indicator for {node_type}.",
            )
        )

    definitions.extend(
        [
            FeatureDefinition(
                name="in_degree",
                value_type=FeatureValueType.COUNT,
                default_value=0.0,
                description="Incoming graph-edge count.",
            ),
            FeatureDefinition(
                name="out_degree",
                value_type=FeatureValueType.COUNT,
                default_value=0.0,
                description="Outgoing graph-edge count.",
            ),
            FeatureDefinition(
                name="total_degree",
                value_type=FeatureValueType.COUNT,
                default_value=0.0,
                description="Total graph-edge count.",
            ),
            FeatureDefinition(
                name="critical_service",
                value_type=FeatureValueType.BINARY,
                default_value=0.0,
                description="Node provides or supports a critical service.",
            ),
            FeatureDefinition(
                name="recent_change_event_count",
                value_type=FeatureValueType.COUNT,
                default_value=0.0,
                description="Visible change events during the observation window.",
            ),
        ]
    )

    for signal_name in SIGNAL_NAMES:
        safe_signal = signal_name.replace(".", "__")

        for statistic in SIGNAL_STATISTICS:
            requires_mask = statistic not in {
                "count",
                "low_quality_fraction",
            }

            definitions.append(
                FeatureDefinition(
                    name=f"{safe_signal}__{statistic}",
                    value_type=(
                        FeatureValueType.COUNT
                        if statistic == "count"
                        else FeatureValueType.CONTINUOUS
                    ),
                    default_value=0.0,
                    requires_mask=requires_mask,
                    description=(
                        f"{statistic} statistic for {signal_name}."
                    ),
                )
            )

        definitions.append(
            FeatureDefinition(
                name=f"{safe_signal}__missing",
                value_type=FeatureValueType.BINARY,
                default_value=1.0,
                description=(
                    f"Missingness indicator for {signal_name}."
                ),
            )
        )

    return definitions


def build_edge_feature_definitions() -> list[FeatureDefinition]:
    definitions = [
        FeatureDefinition(
            name=f"edge_type__{edge_type}",
            value_type=FeatureValueType.CATEGORICAL_ONE_HOT,
            default_value=0.0,
            description=f"One-hot indicator for {edge_type}.",
        )
        for edge_type in EDGE_TYPES
    ]

    definitions.extend(
        [
            FeatureDefinition(
                name="propagation_delay_minutes",
                value_type=FeatureValueType.CONTINUOUS,
                default_value=0.0,
                description="Configured edge-propagation delay.",
            ),
            FeatureDefinition(
                name="propagation_strength",
                value_type=FeatureValueType.CONTINUOUS,
                default_value=1.0,
                description="Configured edge-propagation strength.",
            ),
            FeatureDefinition(
                name="reverse_dependency_direction",
                value_type=FeatureValueType.BINARY,
                default_value=0.0,
                description=(
                    "Whether causal failure propagation runs opposite "
                    "to the stored dependency direction."
                ),
            ),
        ]
    )

    return definitions


class FeatureSchema(BaseModel):
    schema_version: str = "0.1.0"
    node_features: list[FeatureDefinition]
    edge_features: list[FeatureDefinition]

    @model_validator(mode="after")
    def validate_unique_features(self) -> "FeatureSchema":
        node_names = [
            feature.name for feature in self.node_features
        ]
        edge_names = [
            feature.name for feature in self.edge_features
        ]

        if len(node_names) != len(set(node_names)):
            raise ValueError("Duplicate node feature names.")

        if len(edge_names) != len(set(edge_names)):
            raise ValueError("Duplicate edge feature names.")

        return self


def build_feature_schema() -> FeatureSchema:
    return FeatureSchema(
        node_features=build_node_feature_definitions(),
        edge_features=build_edge_feature_definitions(),
    )
