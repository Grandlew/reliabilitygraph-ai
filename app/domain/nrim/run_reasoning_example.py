from __future__ import annotations

import json
from pathlib import Path

from .hypothesis_engine import run_catchup_storage_reasoning
from .telemetry_validation import load_telemetry_batch


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    example_dir = base_dir / "examples"

    telemetry_batch = load_telemetry_batch(
        example_dir / "hotel_180_rooms_telemetry.json"
    )

    result = run_catchup_storage_reasoning(
        case_id="case_hotel_180_operational_001",
        events=telemetry_batch.events,
    )

    output_path = (
        example_dir
        / "catchup_storage_reasoning_output.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result.model_dump(mode="json"),
            file,
            indent=2,
        )

    print(result.conclusion)

    print("\nRanked hypotheses:")

    for hypothesis in result.hypotheses:
        print(
            f"- {hypothesis.name}: "
            f"score={hypothesis.ranking_score}, "
            f"coverage={hypothesis.evidence_coverage}, "
            f"status={hypothesis.status.value}"
        )

    print("\nNext diagnostic tests:")

    for diagnostic in result.ranked_diagnostics:
        print(
            f"- {diagnostic.name}: "
            f"utility={diagnostic.utility_score}"
        )

    print(f"\nSaved output: {output_path}")


if __name__ == "__main__":
    main()
