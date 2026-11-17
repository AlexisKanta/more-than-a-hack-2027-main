import numpy as np

from bsd.degrade import degrade, fill_ball, gap_profile
from bsd.tracking import Tracking

L, W = 105.0, 68.0


def make_tr(n, periods=None, n_home=11, n_away=11, ball=None):
    """Home players spread over x in [5, 45], away over [60, 100]; all observed; ball at centre."""
    rng = np.random.default_rng(123)
    s = n_home + n_away
    team = np.array([0] * n_home + [1] * n_away, dtype=np.int8)
    players = np.empty((n, s, 2), dtype=np.float32)
    players[:, :n_home, 0] = rng.uniform(5, 45, n_home)
    players[:, n_home:, 0] = rng.uniform(60, 100, n_away)
    players[:, :, 1] = rng.uniform(5, 63, s)
    if ball is None:
        ball = np.tile(np.array([[L / 2, W / 2]], dtype=np.float32), (n, 1))
    periods = periods or [(1, 0, n - 1)]
    return Tracking(players, team, ball.astype(np.float32), periods, (L, W))


def make_profile(seed=7, rate=0.41):
    """BAR-ATH-like run lengths: heavy-tailed gaps (median ~270), visible runs scaled to ``rate``."""
    rng = np.random.default_rng(seed)
    gaps = np.maximum(1, rng.lognormal(np.log(270), 1.0, 150)).astype(np.int64)
    vis_mean = gaps.mean() * (1 - rate) / rate
    vis = np.maximum(1, rng.lognormal(np.log(vis_mean) - 0.5, 1.0, 150)).astype(np.int64)
    return {"gap_lengths": gaps, "visible_lengths": vis, "missing_rate": rate,
            "n_tracked": rng.integers(11, 23, 5000)}


def in_period_mask(tr):
    m = np.zeros(len(tr.ball), dtype=bool)
    for _, a, b in tr.periods:
        m[a:b + 1] = True
    return m


def ball_missing_rate(tr):
    inside = in_period_mask(tr)
    return float(np.isnan(tr.ball[inside, 0]).mean())


# --- gap_profile -------------------------------------------------------------------------


def test_gap_profile_runs_and_counts():
    n = 100
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[10:20] = np.nan  # gap of 10 in period 1
    ball[45:60] = np.nan  # straddles the period boundary: 45..49 in period 1, 50..59 in period 2
    ball[95:] = np.nan  # outside both periods: ignored
    tr = make_tr(n, periods=[(1, 0, 49), (2, 50, 89)], ball=ball)
    tr.players[30:40, :5] = np.nan
    p = gap_profile(tr)
    assert sorted(p["gap_lengths"].tolist()) == [5, 10, 10]
    assert sorted(p["visible_lengths"].tolist()) == [10, 25, 30]
    np.testing.assert_allclose(p["missing_rate"], 25 / 90)
    assert len(p["n_tracked"]) == 90
    assert (p["n_tracked"][30:40] == 17).all() and (p["n_tracked"][:30] == 22).all()


# --- fill_ball ---------------------------------------------------------------------------


def test_fill_ball_no_nan_inside_period():
    n = 120
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[5] = (10.0, 20.0)
    ball[15] = (20.0, 30.0)
    ball[30] = (40.0, 10.0)
    # period 1 = 0..39: leading NaN 0..4, gaps, trailing NaN 31..39; period 2 = 40..79 all NaN
    tr = make_tr(n, periods=[(1, 0, 39), (2, 40, 79)], ball=ball)
    before = tr.ball.copy()
    out = fill_ball(tr)
    assert np.isfinite(out.ball[:80]).all()
    np.testing.assert_allclose(out.ball[:5], [[10.0, 20.0]] * 5)  # leading edge held
    np.testing.assert_allclose(out.ball[10], (15.0, 25.0), atol=1e-5)  # linear interpolation
    np.testing.assert_allclose(out.ball[31:40], [[40.0, 10.0]] * 9)  # trailing edge held
    np.testing.assert_allclose(out.ball[40:80], [[L / 2, W / 2]] * 40)  # empty period -> centre
    assert np.isnan(out.ball[80:]).all()  # outside periods untouched
    np.testing.assert_array_equal(tr.ball, before)  # input not mutated


# --- degrade -----------------------------------------------------------------------------


def test_missing_rate():
    profile = make_profile()
    for seed in range(5):
        tr = make_tr(140_000, periods=[(1, 0, 69_999), (2, 70_000, 139_999)])
        out = degrade(tr, profile, seed)
        assert abs(ball_missing_rate(out) - profile["missing_rate"]) <= 0.03


def test_missing_rate_short_sequence():
    """A sequence only a few runs long still meets the rate (adjustment path)."""
    profile = make_profile()
    for seed in range(5):
        out = degrade(make_tr(3000), profile, seed)
        assert abs(ball_missing_rate(out) - profile["missing_rate"]) <= 0.03


def test_mask_independent_of_label():
    """The degraded ball mask must not carry the dead-ball signal.

    The source ball is NaN on exactly the D frames (the sample-game pattern in its purest
    form), so any leak of the source NaN pattern or of the labels would give P(miss|D) ~ 1.
    Stable, not flaky: the seeds are fixed, so the outcome is deterministic, and the pooled
    sample (20 seeds x 100k frames, ~2.5k independent mask cycles) puts the sampling error of the
    difference around 0.01, far inside the 0.05 bound.
    """
    n = 100_000
    labels = np.zeros(n, dtype=np.int8)
    period = 1500
    for start in range(0, n, period):
        labels[start + 1200:start + period] = 2  # 300-frame D block every 60 s
        labels[start:start + 600] = 0
        labels[start + 600:start + 1200] = 1
    ball = np.tile(np.array([[L / 2, W / 2]], dtype=np.float32), (n, 1))
    ball[labels == 2] = np.nan
    profile = make_profile()
    miss_d, miss_alive = [], []
    for seed in range(20):
        out = degrade(make_tr(n, ball=ball), profile, seed)
        miss = np.isnan(out.ball[:, 0])
        miss_d.append(miss[labels == 2])
        miss_alive.append(miss[labels != 2])
    p_d = np.concatenate(miss_d).mean()
    p_alive = np.concatenate(miss_alive).mean()
    assert abs(p_d - p_alive) < 0.05, (p_d, p_alive)


def test_degrade_deterministic():
    profile = make_profile()
    tr = make_tr(5000, periods=[(1, 0, 2499), (2, 2500, 4999)])
    a, b, c = degrade(tr, profile, 0), degrade(tr, profile, 0), degrade(tr, profile, 1)
    np.testing.assert_array_equal(a.players, b.players)
    np.testing.assert_array_equal(a.ball, b.ball)
    assert not np.array_equal(a.ball, c.ball, equal_nan=True)
    assert not np.array_equal(a.players, c.players, equal_nan=True)


def test_degrade_does_not_mutate_input():
    tr = make_tr(3000)
    tr.ball[100:400] = np.nan
    players, ball, team = tr.players.copy(), tr.ball.copy(), tr.team.copy()
    out = degrade(tr, make_profile(), 0)
    np.testing.assert_array_equal(tr.players, players)
    np.testing.assert_array_equal(tr.ball, ball)
    np.testing.assert_array_equal(tr.team, team)
    assert out.players is not tr.players and out.ball is not tr.ball
    assert out.periods == tr.periods and out.pitch == tr.pitch and out.fps == tr.fps


def test_player_drop_follows_n_tracked():
    profile = dict(make_profile(), n_tracked=np.array([10]))
    tr = make_tr(4000)
    tr.players[2000:, 15:] = np.nan  # only 15 observed from frame 2000
    out = degrade(tr, profile, 3)
    counts = np.isfinite(out.players[..., 0]).sum(axis=1)
    assert (counts == 10).all()
    profile["n_tracked"] = np.array([30])  # more than observed: keep everything observed
    counts = np.isfinite(degrade(tr, profile, 3).players[..., 0]).sum(axis=1)
    assert (counts[:2000] == 22).all() and (counts[2000:] == 15).all()


def test_player_drop_count_is_blockwise():
    profile = dict(make_profile(), n_tracked=np.arange(5, 23))
    out = degrade(make_tr(20_000), profile, 0)
    counts = np.isfinite(out.players[..., 0]).sum(axis=1)
    changes = np.count_nonzero(np.diff(counts))
    assert 20_000 / (5 * 25) * 0.5 < changes < 20_000 / 25  # blocks of 1-5 s, not per frame


def test_slots_stay_within_team_and_get_swapped():
    tr = make_tr(30_000)
    profile = dict(make_profile(), n_tracked=np.array([22]))
    out = degrade(tr, profile, 0)
    home, away = out.players[:, tr.team == 0, 0], out.players[:, tr.team == 1, 0]
    assert np.nanmax(home) < 50 and np.nanmin(away) > 55  # columns never change team
    np.testing.assert_array_equal(out.team, tr.team)
    # static synthetic players: a column's position changes only when identities are reassigned
    moved = np.abs(out.players[1:, 0] - out.players[:-1, 0]).max(axis=1) > 3.0
    assert moved.sum() >= 5


def test_jitter_scale_and_smoothness():
    tr = make_tr(10_000)
    profile = dict(make_profile(), n_tracked=np.array([22]))
    out = degrade(tr, profile, 0)
    # undo the slot permutation by matching each output column to its nearest source column
    d = []
    for f in range(0, 10_000, 97):
        src = tr.players[f]
        for p in out.players[f]:
            d.append(np.min(np.abs(src - p).sum(axis=1)))
    err = np.array(d)
    assert 0.1 < np.median(err) < 1.0
    vis = np.isfinite(out.ball[:, 0])
    bnoise = out.ball[vis] - np.array([L / 2, W / 2])
    assert 0.15 < bnoise.std() < 0.45
    # smooth: frame-to-frame steps of the jittered ball are far below the jitter amplitude
    steps = np.abs(np.diff(out.ball[:, 0]))
    assert np.nanmedian(steps) < 0.05
