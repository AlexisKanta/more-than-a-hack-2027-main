"""Tracking container shared by all pipeline stages."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Tracking:
    players: np.ndarray  # (F, S, 2) float32 metres, NaN missing
    team: np.ndarray  # (S,) int8: 0 home, 1 away
    ball: np.ndarray  # (F, 2) float32 metres, NaN missing
    periods: list  # (period, first_index, last_index) inclusive
    pitch: tuple
    fps: int = 25
