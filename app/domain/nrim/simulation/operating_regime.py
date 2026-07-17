from __future__ import annotations

import math
import random
from datetime import datetime

from .models import OperatingRegime


def occupancy_at_time(
    *,
    timestamp: datetime,
    regime: OperatingRegime,
) -> float:
    hour = timestamp.hour

    evening_peak = math.exp(
        -((hour - 21.0) ** 2) / (2.0 * 3.0**2)
    )

    morning_peak = 0.35 * math.exp(
        -((hour - 8.0) ** 2) / (2.0 * 2.0**2)
    )

    multiplier = (
        1.0
        + (regime.evening_peak_multiplier - 1.0)
        * evening_peak
        + morning_peak
    )

    if timestamp.weekday() >= 5:
        multiplier *= regime.weekend_multiplier

    occupancy = regime.base_occupancy_fraction * multiplier

    return max(0.0, min(1.0, occupancy))


def workload_at_time(
    *,
    timestamp: datetime,
    room_count: int,
    regime: OperatingRegime,
    rng: random.Random,
) -> dict[str, float]:
    occupancy = occupancy_at_time(
        timestamp=timestamp,
        regime=regime,
    )

    occupied_rooms = room_count * occupancy

    active_live_streams = max(
        0.0,
        occupied_rooms * 0.55 + rng.gauss(0.0, 2.0),
    )

    active_catchup_sessions = max(
        0.0,
        occupied_rooms * 0.12 + rng.gauss(0.0, 1.0),
    )

    return {
        "occupancy_fraction": occupancy,
        "active_live_streams": active_live_streams,
        "active_catchup_sessions": (
            active_catchup_sessions
        ),
    }
