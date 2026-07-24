from app.domain.nrim.baselines.incident_detector import (
    choose_incident_threshold,
    fit_incident_detector,
    incident_operating_curve,
    incident_detection_score,
)
from tests.domain.nrim.baselines.test_baseline_evaluator import (
    make_window,
)


def test_incident_score_increases_with_error_evidence() -> None:
    healthy = make_window(
        window_id="healthy",
        healthy=True,
        storage_error=0.0,
    )
    incident = make_window(
        window_id="incident",
        healthy=False,
        storage_error=20.0,
    )

    assert incident_detection_score(
        incident
    ) > incident_detection_score(healthy)


def test_incident_threshold_is_selected_separately() -> None:
    windows = [
        make_window(
            window_id=f"healthy_{index}",
            healthy=True,
            storage_error=0.0,
        )
        for index in range(4)
    ] + [
        make_window(
            window_id=f"incident_{index}",
            healthy=False,
            storage_error=20.0,
        )
        for index in range(4)
    ]

    selection = choose_incident_threshold(
        validation_windows=windows,
        maximum_false_selection_rate=0.0,
        minimum_incident_coverage=1.0,
    )

    assert selection.feasible is True
    assert selection.false_selection_rate == 0.0
    assert selection.coverage == 1.0


def test_fitted_detector_and_operating_curve() -> None:
    windows = [
        make_window(
            window_id=f"healthy_{index}",
            healthy=True,
            storage_error=0.0,
        )
        for index in range(4)
    ] + [
        make_window(
            window_id=f"incident_{index}",
            healthy=False,
            storage_error=20.0,
        )
        for index in range(4)
    ]
    model = fit_incident_detector(
        training_windows=windows,
        iterations=100,
    )
    selection = choose_incident_threshold(
        validation_windows=windows,
        model=model,
        maximum_false_selection_rate=0.0,
        minimum_incident_coverage=1.0,
    )

    assert selection.feasible is True
    assert model.predict_score(windows[-1]) > model.predict_score(
        windows[0]
    )
    assert incident_operating_curve(
        windows=windows,
        model=model,
    )
