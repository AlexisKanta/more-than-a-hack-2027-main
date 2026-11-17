import numpy as np

from bsd.tagging import A, D, H, bonus_from_events

N = 1000


def ev(code, s, e, **tags):
    return {"code": code, "start_frame": s, "end_frame": e, "tags": tags}


def run(events, periods=((1, 0, N - 1),)):
    return bonus_from_events(events, N, periods)


def test_set_piece_dead_before():
    b = run([ev("SET PIECES", 500, 550, Team="Barcelona", Type="Throw-in")])
    assert np.all(b[425:500, D] >= 2.0) and b[424, D] == 0 and b[500, D] == 0
    assert np.all(b[500:550, H] == 1.0) and b[550, H] == 0


def test_na_team_no_team_bonus():
    b = run([ev("SET PIECES", 500, 550, Team="N/A", Type="Throw-in")])
    assert b[:, H].sum() == 0 and b[:, A].sum() == 0 and b[:, D].sum() > 0


def test_case_insensitive_codes():
    up = run([ev("SET PIECES", 500, 550, Team="Atletic Club", Type="Corner Kick")])
    lo = run([ev("set pieces", 500, 550, Team="atletic club", Type="corner kick")])
    assert np.array_equal(up, lo) and up[500, A] == 1.0


def test_goal_dead_until_kickoff():
    evs = [ev("GOALS", 100, 120, Team="Barcelona"),
           ev("SET PIECES", 300, 320, Team="Atletic Club", Type="Kick Off (After Goal)")]
    b = run(evs, periods=((1, 0, 0),))  # keep R14 out of the way
    assert np.all(b[120:300, D] >= 2.0) and b[119, D] == 0
    assert b[300 - 76, D] == 2.0 and b[310, D] == 0  # past the kickoff only team bonus
    capped = run([ev("GOALS", 100, 120, Team="Barcelona")], periods=((1, 0, 0),))
    assert capped[120 + 2249, D] == 2.0 and capped[120 + 2250:, D].sum() == 0 if N > 2400 else True
    assert capped[:, D].sum() == 2.0 * (N - 120)


def test_team_phase_bonus():
    b = run([ev("BUILD UP", 10, 60, Team="Barcelona"), ev("BUILD UP", 70, 80, Team="N/A")])
    assert np.allclose(b[10:60, H], 0.3) and b[70:80].sum() == 0


def test_pre_kickoff_dead_per_period():
    evs = [ev("SET PIECES", 200, 250, Team="Barcelona", Type="Kick Off (Start)"),
           ev("SET PIECES", 700, 750, Team="Atletic Club", Type="Kick Off (Start)"),
           ev("SET PIECES", 50, 60, Team="N/A", Type="Throw-in")]
    b = run(evs, periods=((1, 100, 400), (2, 600, 900)))
    assert np.all(b[100:200, D] >= 3.0) and np.all(b[600:700, D] >= 3.0)
    assert b[99, D] == 0 and b[300, D] == 0 and b[500, D] == 0


def test_r14b_start_tag_before_period_gives_no_bonus():
    evs = [ev("SET PIECES", 90, 130, Team="Barcelona", Type="Kick Off (Start)"),
           ev("SET PIECES", 600, 640, Team="Atletic Club", Type="Kick Off (After Goal)")]
    b = run(evs, periods=((1, 100, 900),))
    assert b[100:520, D].max() == 0.0


def test_r14b_after_goal_never_triggers():
    evs = [ev("SET PIECES", 600, 640, Team="Atletic Club", Type="Kick Off (After Goal)")]
    b = run(evs, periods=((1, 100, 900),))
    assert b[100:520, D].max() == 0.0


def test_goal_cap_inside_range():
    n = 5000
    b = bonus_from_events([ev("GOALS", 100, 120, Team="Barcelona")], n, ((1, 0, 0),))
    assert b[120 + 2249, D] == 2.0 and b[120 + 2250, D] == 0.0
