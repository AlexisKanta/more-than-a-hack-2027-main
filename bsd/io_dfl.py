"""Reader for the public DFL/IDSSE Bundesliga positions XML (Bassek et al., Sci Data 2025, CC BY 4.0).

Positions files hold one ``FrameSet`` per person per ``GameSection`` (firstHalf, secondHalf, ...),
each a run of ``<Frame N= X= Y= .../>``. The ball FrameSet (TeamId "BALL") additionally carries
``BallPossession`` (1 home, 2 away) and ``BallStatus`` (1 alive, 0 dead), which give the labels.

Coordinates: DFL uses metres with the origin at the pitch centre (X -L/2..L/2, Y -W/2..W/2). We add
(L/2, W/2) so the origin is a pitch corner (0..L, 0..W), as in ``bsd.io_metrica_csv``. No axis is
flipped (Y direction is kept as delivered), and attacking direction per half is left as delivered.

Slots: one per (team, PersonId), so a player keeps one column across both halves (NaN while off the
pitch or not yet on). Referees are dropped. Frame index: halves are concatenated in kick-off order,
each spanning the first to last DFL frame number N seen in that GameSection; the gap between halves
is not part of the index.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from bsd.fetch_dfl import DFL_FILES, dfl_dir, verify
from bsd.tracking import Tracking

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

H, A, D, X = 0, 1, 2, 3  # label codes, as in bsd.decode


def read_matchinfo(path):
    """Return (home_team_id, (pitch_length_m, pitch_width_m))."""
    root = ET.parse(str(path)).getroot()
    home = None
    for team in root.iter("Team"):
        if team.get("Role") == "home":
            home = team.get("TeamId")
    env = root.find(".//Environment")
    if home is None or env is None:
        raise ValueError("matchinformation lacks home team or Environment: %s" % path)
    return home, (float(env.get("PitchX")), float(env.get("PitchY")))


def _is_person(team_id):
    return (team_id or "").lower() not in ("ball", "referee")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _stream(path):
    """Yield one record per FrameSet: (section, team_id, person_id, N, x, y, status, poss)."""
    n, x, y, st, po = [], [], [], [], []
    for _, el in ET.iterparse(str(path), events=("end",)):
        if el.tag == "Frame":
            a = el.attrib
            n.append(int(a["N"]))
            x.append(_num(a.get("X")))
            y.append(_num(a.get("Y")))
            st.append(int(a.get("BallStatus", -1)))
            po.append(int(a.get("BallPossession", -1)))
            el.clear()
        elif el.tag == "FrameSet":
            yield (
                el.get("GameSection"), el.get("TeamId"), el.get("PersonId"),
                np.array(n, np.int64), np.array(x, np.float32), np.array(y, np.float32),
                np.array(st, np.int8), np.array(po, np.int8),
            )
            n, x, y, st, po = [], [], [], [], []
            el.clear()


def read_dfl(positions_xml, matchinfo_xml):
    """Parse one match into ``(Tracking, labels)``; labels is int8 (F,) with H=0 A=1 D=2 X=3."""
    home_id, pitch = read_matchinfo(matchinfo_xml)
    half = np.array(pitch, np.float32) / 2
    sets = [s for s in _stream(positions_xml) if s[3].size]
    ball_sets = [s for s in sets if (s[1] or "").lower() == "ball"]
    sections = {}
    for s in sets:
        lo, hi = int(s[3].min()), int(s[3].max())
        a, b = sections.get(s[0], (lo, hi))
        sections[s[0]] = (min(a, lo), max(b, hi))
    order = sorted(sections, key=lambda k: sections[k][0])
    offset, periods = {}, []
    total = 0
    for p, sec in enumerate(order, 1):
        lo, hi = sections[sec]
        offset[sec] = total - lo
        periods.append((p, total, total + hi - lo))
        total += hi - lo + 1

    keys = []
    for s_ in sets:
        if _is_person(s_[1]):
            k = (0 if s_[1] == home_id else 1, s_[2])
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: k[0])  # home first; stable within team
    slot_of = {k: i for i, k in enumerate(keys)}
    team = [k[0] for k in keys]

    players = np.full((total, len(keys), 2), np.nan, np.float32)
    ball = np.full((total, 2), np.nan, np.float32)
    labels = np.full(total, X, np.int8)
    for sec, tid, pid, n, x, y, st, po in sets:
        idx = n + offset[sec]
        xy = np.stack([x, y], axis=1) + half
        if (tid or "").lower() == "ball":
            ball[idx] = xy
            lab = np.full(len(n), X, np.int8)
            lab[po == 1] = H
            lab[po == 2] = A
            lab[st == 0] = D
            labels[idx] = lab
        elif _is_person(tid):
            players[idx, slot_of[(0 if tid == home_id else 1, pid)]] = xy
    if not ball_sets:
        raise ValueError("no ball FrameSet in %s" % positions_xml)
    trk = Tracking(players, np.array(team, np.int8), ball, periods, (float(pitch[0]), float(pitch[1])))
    return trk, labels


def _files(match_id):
    short = match_id.replace("DFL-MAT-", "")
    out = {}
    for _, name, md5 in DFL_FILES:
        if name.endswith("_DFL-MAT-%s.xml" % short):
            if name.startswith("DFL_02_01_"):
                out["matchinformation"] = (dfl_dir() / name, md5)
            elif name.startswith("DFL_04_03_"):
                out["positions"] = (dfl_dir() / name, md5)
    if len(out) != 2:
        raise KeyError("unknown DFL match id: %s" % match_id)
    return short, out


def dfl_game(match_id, cache_dir=None):
    """``(Tracking, labels)`` for a DFL match id such as ``"J03WMX"``, cached to cache/dfl_<id>.npz."""
    short, files = _files(match_id)
    cache = Path(cache_dir or CACHE_DIR) / ("dfl_%s.npz" % short)
    if cache.exists():
        z = np.load(cache)
        periods = [tuple(int(v) for v in r) for r in z["periods"]]
        trk = Tracking(z["players"], z["team"], z["ball"], periods, tuple(float(v) for v in z["pitch"]))
        return trk, z["labels"]
    for path, md5 in files.values():
        verify(path, md5)
    trk, labels = read_dfl(files["positions"][0], files["matchinformation"][0])
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, players=trk.players, team=trk.team, ball=trk.ball, labels=labels,
             periods=np.array(trk.periods, np.int64), pitch=np.array(trk.pitch, np.float64))
    return trk, labels
