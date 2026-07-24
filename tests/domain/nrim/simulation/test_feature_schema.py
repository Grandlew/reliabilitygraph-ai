from app.domain.nrim.simulation.feature_schema import (
    SIGNAL_NAMES,
    build_feature_schema,
)


def test_feature_names_are_unique() -> None:
    schema = build_feature_schema()

    node_names = [
        feature.name
        for feature in schema.node_features
    ]

    edge_names = [
        feature.name
        for feature in schema.edge_features
    ]

    assert len(node_names) == len(set(node_names))
    assert len(edge_names) == len(set(edge_names))


def test_every_signal_has_missingness_feature() -> None:
    schema = build_feature_schema()

    node_names = {
        feature.name
        for feature in schema.node_features
    }

    for signal_name in SIGNAL_NAMES:
        safe_signal = signal_name.replace(
            ".",
            "__",
        )

        assert (
            f"{safe_signal}__missing"
            in node_names
        )
        assert (
            f"{safe_signal}__applicable"
            in node_names
        )


def test_feature_schema_contains_node_types() -> None:
    schema = build_feature_schema()

    names = {
        feature.name
        for feature in schema.node_features
    }

    assert "node_type__catchup_storage" in names
    assert "node_type__middleware" in names
