import numpy as np

from bsd.decode import A, D, H
from bsd.features import mirror
from bsd.heuristic import heuristic_logp

NAMES = [
    "ball_visible", "ball_out", "ball_speed", "ball_dist_edge", "near_home", "near_away",
    "n2_home", "n2_away", "ball_static", "depth_diff_m3s", "vx_fwd_home_m3s", "vx_fwd_away_m3s",
    "speed_all_m1s", "ball_speed_m1s",
]


def _rows(**cols):
    """One frame per entry: unspecified columns are NaN (undefined)."""
    n = len(next(iter(cols.values())))
    X = np.full((n, len(NAMES)), np.nan, dtype=np.float32)
    for k, v in cols.items():
        X[:, NAMES.index(k)] = v
    return X


def test_shape_and_finite_on_all_nan():
    X = np.full((5, len(NAMES)), np.nan, dtype=np.float32)
    lp = heuristic_logp(X, NAMES)
    assert lp.shape == (5, 3)
    assert np.isfinite(lp).all()
    assert np.allclose(lp, 0)


def test_controlling_player_gives_team():
    base = dict(ball_visible=[1, 1], ball_out=[0, 0], ball_speed=[3, 3], ball_dist_edge=[20, 20],
                speed_all_m1s=[2, 2])
    X = _rows(near_home=[0.5, 6.0], near_away=[6.0, 0.5], n2_home=[1, 0], n2_away=[0, 1], **base)
    lp = heuristic_logp(X, NAMES)
    assert lp[0].argmax() == H and lp[1].argmax() == A


def test_ball_out_gives_dead():
    X = _rows(ball_visible=[1], ball_out=[1], ball_speed=[1], ball_dist_edge=[-3], near_home=[1],
              near_away=[3], n2_home=[1], n2_away=[0], speed_all_m1s=[1.5])
    assert heuristic_logp(X, NAMES)[0].argmax() == D


def test_walking_players_without_ball_give_dead():
    X = _rows(ball_visible=[0, 0], speed_all_m1s=[0.5, 2.5])
    lp = heuristic_logp(X, NAMES)
    assert lp[0].argmax() == D
    assert lp[1, D] < max(lp[1, H], lp[1, A])


def test_magnitudes_modest():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 20, (2000, len(NAMES))).astype(np.float32)
    assert np.abs(heuristic_logp(X, NAMES)).max() <= 3.5


def test_team_symmetry_under_mirror():
    rng = np.random.default_rng(1)
    names = NAMES + ["depth_diff"]  # mirror needs every _home/_away pair; _diff is negated
    X = np.abs(rng.normal(2, 2, (500, len(names)))).astype(np.float32)
    X[:, names.index("depth_diff_m3s")] -= 2
    a = heuristic_logp(X, names)
    b = heuristic_logp(mirror(X, names), names)
    assert np.allclose(a[:, [H, A, D]], b[:, [A, H, D]], atol=1e-6)
