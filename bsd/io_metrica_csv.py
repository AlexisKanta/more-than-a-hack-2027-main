"""Reader for Metrica Sports sample-data raw tracking CSVs (Sample Games 1-2)."""
import numpy as np
import pandas as pd

from bsd.tracking import Tracking

_HEADER_ROWS = 3  # team names, jersey numbers, column names
_META_COLS = 3  # Period, Frame, Time [s]


def _read(path):
    # Positional columns only: the y columns are unnamed in the header.
    return pd.read_csv(path, skiprows=_HEADER_ROWS, header=None, dtype=np.float64)


def _periods(period, frame):
    idx = frame.astype(np.int64) - 1
    out = []
    start = 0
    for i in range(1, len(period) + 1):
        if i == len(period) or period[i] != period[start]:
            out.append((int(period[start]), int(idx[start]), int(idx[i - 1])))
            start = i
    return out


def read_metrica_csv(home_csv, away_csv, pitch=(105.0, 68.0)):
    """Parse a Home/Away CSV pair into a ``Tracking`` (metres, index = Frame - 1).

    Ball is taken from the home file (last column pair). Coordinates are only scaled
    (x * L, y * W); no axis flips.
    """
    home, away = _read(home_csv), _read(away_csv)
    if len(home) != len(away):
        raise ValueError("home and away files have different frame counts")
    hv = home.to_numpy()
    av = away.to_numpy()
    n_home = (hv.shape[1] - _META_COLS - 2) // 2
    n_away = (av.shape[1] - _META_COLS - 2) // 2
    xy = np.concatenate(
        [hv[:, _META_COLS:_META_COLS + 2 * n_home], av[:, _META_COLS:_META_COLS + 2 * n_away]],
        axis=1,
    )
    players = xy.reshape(len(hv), n_home + n_away, 2)
    ball = hv[:, _META_COLS + 2 * n_home:_META_COLS + 2 * n_home + 2]
    scale = np.array(pitch, dtype=np.float64)
    team = np.concatenate([np.zeros(n_home, np.int8), np.ones(n_away, np.int8)])
    return Tracking(
        players=(players * scale).astype(np.float32),
        team=team,
        ball=(ball * scale).astype(np.float32),
        periods=_periods(hv[:, 0], hv[:, 1]),
        pitch=tuple(pitch),
    )
