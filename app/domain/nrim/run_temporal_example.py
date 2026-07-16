from __future__ import annotations

import json
from pathlib import Path

from .precursor_detector import (
    detect_catchup_storage_precursor,
)
from .storage_forecast import forecast_threshold_crossing
from .temporal import TemporalSeries
from .temporal_backtest import rolling_origin_backtest
from .telemetry_validation import load_telemetry_batch


def load_temporal_series(path: Path) -> TemporalSeries:
    with path.open("r", encoding="utf-8") as file:
        return TemporalSeries.model_validate(json.load(file))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    example_dir = base_dir / "examples"

    complete_series = load_temporal_series(
        example_dir / "catchup_storage_timeseries.json"
    )

    current_series = complete_series.model_copy(
        update={
            "points": complete_series.points[:18],
        }
    )

    forecast = forecast_threshold_crossing(
        current_series,
        threshold=95.0,
    )

    telemetry_batch = load_telemetry_batch(
        example_dir / "hotel_180_rooms_telemetry.json"
    )

    warning = detect_catchup_storage_precursor(
        case_id="case_hotel_180_temporal_001",
        forecast=forecast,
        events=telemetry_batch.events,
    )

    backtest = rolling_origin_backtest(
        complete_series,
        threshold=95.0,
        minimum_training_points=12,
    )

    output = {
        "forecast": forecast.model_dump(mode="json"),
        "warning": warning.model_dump(mode="json"),
        "backtest": backtest.model_dump(mode="json"),
        "limitations": [
            "All example data is synthetic.",
            "Forecast intervals are not statistically calibrated.",
            "Thresholds are provisional.",
            "No real NetUP incident outcome is used."
        ]
    }

    output_path = (
        example_dir / "catchup_early_warning_output.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("Forecast status:", forecast.status.value)

    if forecast.central_hours_to_threshold is not None:
        print(
            "Central hours to threshold:",
            forecast.central_hours_to_threshold,
        )

    print("Warning tier:", warning.tier.value)
    print("Warning conclusion:", warning.conclusion)

    print(
        "Backtest evaluations:",
        backtest.evaluation_count,
    )
    print(
        "Backtest MAE hours:",
        backtest.mean_absolute_error_hours,
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
