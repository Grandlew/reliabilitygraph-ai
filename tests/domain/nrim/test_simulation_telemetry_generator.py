import random

import pytest

from app.domain.nrim.simulation.telemetry_generator import (
    sample_poisson,
)


def test_poisson_zero_rate_returns_zero() -> None:
    assert sample_poisson(random.Random(1), 0.0) == 0


@pytest.mark.parametrize("rate", [-0.1, float("inf"), float("nan")])
def test_poisson_rejects_invalid_rate(rate: float) -> None:
    with pytest.raises(ValueError):
        sample_poisson(random.Random(1), rate)


def test_poisson_is_reproducible() -> None:
    first_rng = random.Random(42)
    second_rng = random.Random(42)

    first = [sample_poisson(first_rng, 1.1) for _ in range(100)]
    second = [sample_poisson(second_rng, 1.1) for _ in range(100)]

    assert first == second


def test_poisson_sample_mean_tracks_rate() -> None:
    rng = random.Random(42)
    rate = 1.1
    samples = [sample_poisson(rng, rate) for _ in range(20_000)]

    assert sum(samples) / len(samples) == pytest.approx(
        rate,
        abs=0.03,
    )
