import numpy as np
import pytest

from bsd import fetch_dfl, io_dfl

INFO = """<?xml version="1.0"?>
<PutDataRequest><MatchInformation>
<Environment PitchX="100.00" PitchY="60.00" />
<Teams>
<Team TeamId="TAWAY" Role="guest"><Players/></Team>
<Team TeamId="THOME" Role="home"><Players/></Team>
</Teams></MatchInformation></PutDataRequest>
"""

# half -> (first N, ball rows: (status, possession)); gap of unrelated N between the halves
_BALL = {
    "firstHalf": (100, [(1, 1), (1, 2), (0, 1), (1, 0)]),
    "secondHalf": (500, [(0, 2), (1, 2), (1, 1)]),
}


def _fs(section, team, person, n0, rows):
    out = ['<FrameSet GameSection="%s" MatchId="M" TeamId="%s" PersonId="%s">' % (section, team, person)]
    for i, r in enumerate(rows):
        out.append('<Frame N="%d" T="t" %s/>' % (n0 + i, r))
    out.append("</FrameSet>")
    return "\n".join(out)


def _positions():
    parts = ['<?xml version="1.0"?>', "<PutDataRequest><Positions>"]
    for sec, (n0, ball) in _BALL.items():
        n = len(ball)
        parts.append(_fs(sec, "referee", "R1", n0, ['X="1.00" Y="1.00"'] * n))
        for team, pers in (("THOME", ("P1", "P2")), ("TAWAY", ("Q1", "Q2"))):
            for j, pid in enumerate(pers):
                rows = ['X="%.2f" Y="%.2f"' % (j + 1, -j - 1)] * n
                if pid == "P1" and sec == "firstHalf":
                    rows[1] = 'X="-50.00" Y="-30.00"'  # known point -> (0, 0)
                    rows[2] = 'X="NaN" Y="NaN"'
                parts.append(_fs(sec, team, pid, n0, rows))
        rows = ['X="0.5" Y="-0.5" Z="0.00" BallPossession="%d" BallStatus="%d"' % (po, st) for st, po in ball]
        parts.append(_fs(sec, "BALL", "DFL-OBJ-0000BALL", n0, rows))
    parts.append("</Positions></PutDataRequest>")
    return "\n".join(parts)


@pytest.fixture
def parsed(tmp_path):
    pos, info = tmp_path / "pos.xml", tmp_path / "info.xml"
    pos.write_text(_positions())
    info.write_text(INFO)
    return io_dfl.read_dfl(pos, info)


def test_shapes_team_periods(parsed):
    trk, labels = parsed
    assert trk.players.shape == (7, 4, 2)
    assert trk.ball.shape == (7, 2)
    assert labels.shape == (7,) and labels.dtype == np.int8
    assert trk.team.tolist() == [0, 0, 1, 1]
    assert trk.periods == [(1, 0, 3), (2, 4, 6)]  # gap between halves is not indexed
    assert trk.pitch == (100.0, 60.0) and trk.fps == 25


def test_coordinates_corner_origin(parsed):
    trk, _ = parsed
    # centre-origin (-50, -30) is the pitch corner (0, 0); (0.5, -0.5) is the centre offset
    np.testing.assert_allclose(trk.players[1, 0], [0.0, 0.0])
    np.testing.assert_allclose(trk.ball[0], [50.5, 29.5])
    np.testing.assert_allclose(trk.players[0, 1], [52.0, 28.0])  # P2: (2, -2)
    assert np.isnan(trk.players[2, 0]).all()  # NaN stays missing


def test_labels(parsed):
    _, labels = parsed
    # status 0 -> D; else possession 1 -> H, 2 -> A, otherwise X
    assert labels.tolist() == [0, 1, 2, 3, 2, 1, 0]


def test_fetch_md5_check_raises(tmp_path, monkeypatch):
    def fake(url, path):
        with open(path, "wb") as f:
            f.write(b"corrupted")

    monkeypatch.setattr(fetch_dfl, "_download", fake)
    with pytest.raises(ValueError, match="MD5"):
        fetch_dfl.fetch(tmp_path, matches=["J03WMX"])
    assert not list(tmp_path.glob("*.xml"))  # corrupt download is not kept


def test_fetch_skips_existing_and_lists_files(tmp_path, monkeypatch):
    assert len(fetch_dfl.DFL_FILES) == 21
    calls = []
    monkeypatch.setattr(fetch_dfl, "_download", lambda u, p: calls.append(u))
    for _, name, _ in fetch_dfl.DFL_FILES:
        (tmp_path / name).write_text("x")
    fetch_dfl.fetch(tmp_path)
    assert calls == []
