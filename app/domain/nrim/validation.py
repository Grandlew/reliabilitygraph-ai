from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import ReliabilityCase


def load_reliability_case(path: Path) -> ReliabilityCase:
    if not path.exists():
        raise FileNotFoundError(f"Reliability case not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_case = json.load(file)

    return ReliabilityCase.model_validate(raw_case)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    example_path = base_dir / "examples" / "hotel_180_rooms_case.json"

    try:
        case = load_reliability_case(example_path)
    except ValidationError as error:
        print("Reliability case is invalid:")
        print(error)
        raise SystemExit(1)

    print(f"Valid reliability case: {case.case_id}")
    print(f"Nodes: {len(case.nodes)}")
    print(f"Edges: {len(case.edges)}")
