import numpy as np

from bsd.io_metrica_csv import read_metrica_csv


def _write(path, team, jerseys, rows):
    n = len(jerseys)
    head1 = ",,," + ",".join([team, ""] * n) + ",,"
    head2 = ",,," + ",".join(f"{j}," for j in jerseys) + ",,"
    head3 = "Period,Frame,Time [s]," + ",".join(f"Player{j}," for j in jerseys) + "Ball,"
    lines = [head1, head2, head3] + [",".join(str(v) for v in r) + "," for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_metrica_csv_shapes(tmp_path):
    nan = "NaN"
    home = tmp_path / "h.csv"
    away = tmp_path / "a.csv"
    # (period, frame, time, p1x, p1y, p2x, p2y, ballx, bally)
    _write(home, "Home", [1, 2], [
        (1, 1, 0.04, 0.5, 0.5, 0.1, 0.2, 0.25, 0.75),
        (1, 2, 0.08, 0.6, 0.5, nan, nan, nan, nan),
        (2, 3, 0.12, 0.0, 1.0, 0.2, 0.4, 1.0, 0.0),
    ])
    _write(away, "Away", [3], [
        (1, 1, 0.04, 0.9, 0.9, 0.5, 0.5),
        (1, 2, 0.08, 0.8, 0.1, 0.5, 0.5),
        (2, 3, 0.12, nan, nan, 0.5, 0.5),
    ])
    t = read_metrica_csv(home, away, pitch=(100.0, 50.0))
    assert t.players.shape == (3, 3, 2)
    assert t.ball.shape == (3, 2)
    assert t.team.tolist() == [0, 0, 1]
    assert t.players.dtype == np.float32 and t.ball.dtype == np.float32
    assert t.players[0, 0].tolist() == [50.0, 25.0]
    assert t.players[0, 1].tolist() == [10.0, 10.0]
    assert t.players[0, 2].tolist() == [90.0, 45.0]
    assert np.isnan(t.players[1, 1]).all() and np.isnan(t.players[2, 2]).all()
    assert t.ball[0].tolist() == [25.0, 37.5]
    assert np.isnan(t.ball[1]).all()
    assert t.ball[2].tolist() == [100.0, 0.0]
    assert t.periods == [(1, 0, 1), (2, 2, 2)]
    assert t.pitch == (100.0, 50.0) and t.fps == 25
