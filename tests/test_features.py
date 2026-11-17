import warnings

import numpy as np

from bsd.features import attack_sign, frame_features, mirror
from bsd.tracking import Tracking

L, W = 105.0, 68.0


def make_tr(n=200, periods=None, ball=None, home_x=30.0, away_x=75.0):
    """Two home slots (0, 1) and two away slots (2, 3) standing still, ball at the centre spot."""
    team = np.array([0, 0, 1, 1], dtype=np.int8)
    players = np.zeros((n, 4, 2), dtype=np.float32)
    players[:, 0] = (home_x, 30.0)
    players[:, 1] = (home_x, 38.0)
    players[:, 2] = (away_x, 30.0)
    players[:, 3] = (away_x, 38.0)
    if ball is None:
        ball = np.tile(np.array([[L / 2, W / 2]], dtype=np.float32), (n, 1))
    periods = periods or [(1, 0, n - 1)]
    return Tracking(players, team, ball.astype(np.float32), periods, (L, W))


def col(X, names, name):
    return X[:, names.index(name)]


def test_ball_out_flag():
    tr = make_tr(n=100)
    tr.ball[:50] = (-1.0, 20.0)
    X, names = frame_features(tr)
    out = col(X, names, "ball_out")
    assert (out[:50] == 1).all()
    assert (out[50:] == 0).all()


def test_near_distances():
    tr = make_tr(n=50)
    tr.players[:, 0] = (L / 2 + 1.0, W / 2)  # home 1 m from the ball
    tr.players[:, 2] = (L / 2, W / 2 - 3.0)  # away 3 m from the ball
    X, names = frame_features(tr)
    np.testing.assert_allclose(col(X, names, "near_home"), 1.0, atol=1e-4)
    np.testing.assert_allclose(col(X, names, "near_away"), 3.0, atol=1e-4)
    assert (col(X, names, "n2_home") == 1).all() and (col(X, names, "n2_away") == 0).all()
    assert (col(X, names, "n5_home") == 1).all() and (col(X, names, "n5_away") == 1).all()


BALL_COLS = ("ball_out", "ball_speed", "ball_dist_edge", "near_home", "near_away", "n2_home",
             "n2_away", "n5_home", "n5_away", "frames_since_ball", "ball_static")


def test_missing_ball_nan_safe():
    tr = make_tr(n=120, ball=np.full((120, 2), np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X, names = frame_features(tr)
    assert X.dtype == np.float32 and X.shape == (120, len(names))
    assert (col(X, names, "ball_visible") == 0).all()
    for c in BALL_COLS:
        assert np.isnan(col(X, names, c)).all(), c
    # shape features are still available without the ball
    assert np.isfinite(col(X, names, "cx_home")).all()


def random_tr(seed=0, n=600):
    """Moving players with gaps and an identity jump, two periods with a direction flip."""
    rng = np.random.default_rng(seed)
    s = 10
    team = np.array([0] * 5 + [1] * 5, dtype=np.int8)
    base = np.where(team[:, None] == 0, [[30.0, 34.0]], [[70.0, 34.0]]) + rng.normal(0, 5, (s, 2))
    walk = np.cumsum(rng.normal(0, 0.1, (n, s, 2)), axis=0)
    players = (base[None] + walk).astype(np.float32)
    players[n // 2 + 10:] = (np.array([L, W]) - players[n // 2 + 10:])  # 2nd half: sides swapped
    players[rng.random((n, s)) < 0.1] = np.nan
    players[100:, 3] += 15.0  # identity jump
    ball = (np.array([L / 2, W / 2]) + np.cumsum(rng.normal(0, 0.5, (n, 2)), axis=0)).astype(np.float32)
    ball[rng.random(n) < 0.3] = np.nan
    ball[200:260] = np.nan
    periods = [(1, 5, n // 2), (2, n // 2 + 10, n - 3)]
    return Tracking(players, team, ball, periods, (L, W))


def test_mirror_swaps():
    tr = random_tr()
    X, names = frame_features(tr)
    sw = Tracking(tr.players, (1 - tr.team).astype(np.int8), tr.ball, tr.periods, tr.pitch, tr.fps)
    Xs, names_s = frame_features(sw)
    assert names_s == names
    np.testing.assert_allclose(mirror(X, names), Xs, atol=1e-5, equal_nan=True)
    # mirror is an involution
    np.testing.assert_array_equal(mirror(mirror(X, names), names), X)


def test_mirror_name_rules():
    names = ["near_home", "near_away", "depth_diff_m1s", "speed_all", "n2_home_m3s", "n2_away_m3s"]
    X = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(mirror(X, names), [[2, 1, -3, 4, 6, 5]])


def test_attack_direction_per_period():
    tr = make_tr(n=200, periods=[(1, 0, 99), (2, 100, 199)])
    tr.players[100:, :2, 0] = 75.0
    tr.players[100:, 2:, 0] = 30.0
    s = attack_sign(tr)
    assert (s[:100] == 1).all() and (s[100:] == -1).all()


def test_vx_fwd_follows_attack_direction():
    tr = make_tr(n=200, periods=[(1, 0, 99), (2, 100, 199)])
    tr.players[100:, :2, 0] = 75.0
    tr.players[100:, 2:, 0] = 30.0
    tr.players[:, :2, 0] += np.arange(200, dtype=np.float32)[:, None] * 0.04  # home moves +x, 1 m/s
    X, names = frame_features(tr)
    v = col(X, names, "vx_fwd_home")
    np.testing.assert_allclose(v[1:100], 1.0, atol=1e-3)  # attacks +x: forward
    np.testing.assert_allclose(v[101:200], -1.0, atol=1e-3)  # attacks -x: backward
    assert np.isnan(v[0]) and np.isnan(v[100])  # no velocity across a period boundary
    np.testing.assert_allclose(col(X, names, "vx_fwd_away")[1:100], 0.0, atol=1e-6)


def test_velocity_ignores_identity_jumps():
    tr = make_tr(n=60)
    tr.players[30:, 0, 0] += 20.0  # slot 0 jumps 20 m in one frame
    X, names = frame_features(tr)
    sp = col(X, names, "speed_home")
    np.testing.assert_allclose(sp[1:], 0.0, atol=1e-6)


def test_frames_since_ball_resets_per_period():
    n = 100
    ball = np.tile(np.array([[L / 2, W / 2]], dtype=np.float32), (n, 1))
    ball[10:30] = np.nan
    ball[50:55] = np.nan  # start of period 2: nothing seen yet
    tr = make_tr(n=n, periods=[(1, 0, 49), (2, 50, 99)], ball=ball)
    X, names = frame_features(tr)
    f = col(X, names, "frames_since_ball")
    assert f[9] == 0 and f[10] == 1 and f[29] == 20 and f[30] == 0
    assert np.isnan(f[50:55]).all() and f[55] == 0


def test_rolling_means_do_not_cross_periods():
    n = 200
    tr = make_tr(n=n, periods=[(1, 0, 99), (2, 100, 199)])
    tr.ball[:100] = (-5.0, 20.0)  # out for all of period 1, in for period 2
    X, names = frame_features(tr)
    m = col(X, names, "ball_out_m3s")
    assert (m[:100] == 1).all() and (m[100:] == 0).all()


def test_outside_periods_nan_and_dtype():
    tr = make_tr(n=100, periods=[(1, 10, 89)])
    X, names = frame_features(tr)
    assert X.dtype == np.float32 and len(set(names)) == len(names)
    assert np.isnan(X[:10]).all() and np.isnan(X[90:]).all()
    assert np.isfinite(col(X, names, "speed_all")[11:90]).all()
