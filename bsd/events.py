"""Sample-game event logs -> per-frame possession labels (H/A/D/X)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bsd.io_epts import _parse_meta, _periods_from_params, read_epts
from bsd.io_metrica_csv import read_metrica_csv
from bsd.tracking import Tracking

# Same codes as evaluation/possession_metrics.py (H, A, D, X = 0, 1, 2, 3).
H, A, D, X = 0, 1, 2, 3

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample"
CACHE_DIR = ROOT / "cache"
PITCH = (105.0, 68.0)

OWNER_TYPES = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}
CONTROL_TYPES = {"PASS", "CARRY", "RECOVERY", "SHOT"}
COLUMNS = ["team", "type", "subtype", "period", "start", "end"]


def _finish(df):
    df = pd.DataFrame(df, columns=COLUMNS)
    for c in ("period", "start", "end"):
        df[c] = df[c].astype(np.int64)
    return df.sort_values("start", kind="stable").reset_index(drop=True)


def load_events_csv(path):
    """Metrica sample-data raw events CSV (Games 1-2). Index = frame - 1; End Frame 0 -> start."""
    raw = pd.read_csv(path)
    start = raw["Start Frame"].astype(np.int64) - 1
    end = raw["End Frame"].astype(np.int64) - 1
    end = end.where(raw["End Frame"] > 0, start)
    return _finish({
        "team": raw["Team"].map({"Home": "H", "Away": "A"}),
        "type": raw["Type"].str.upper().str.strip(),
        "subtype": raw["Subtype"].fillna("").str.upper().str.strip(),
        "period": raw["Period"],
        "start": start,
        "end": end,
    })


def _subtype_str(st):
    if not st:
        return ""
    if isinstance(st, dict):
        st = [st]
    return "-".join(str(s["name"]).upper().strip() for s in st)


def load_events_json(path, home_team_id):
    """FIFA/EPTS-style events JSON (Game 3). Index = frame - 1."""
    with open(path, "r") as fh:
        data = json.load(fh)["data"]
    rows = []
    for e in data:
        s = int(e["start"]["frame"]) - 1
        end_f = e["end"]["frame"] if e.get("end") else None
        rows.append({
            "team": "H" if e["team"]["id"] == home_team_id else "A",
            "type": e["type"]["name"].upper().strip(),
            "subtype": _subtype_str(e.get("subtypes")),
            "period": e["period"],
            "start": s,
            "end": int(end_f) - 1 if end_f is not None else s,
        })
    return _finish(rows)


def _is_restart(typ, sub):
    return typ == "SET PIECE" or "GOAL KICK" in sub


def _is_foul(typ, sub):
    return typ == "FAULT RECEIVED" or (typ == "CHALLENGE" and "FAULT" in sub)


def _is_trigger(typ, sub):
    if typ in ("BALL OUT", "FAULT RECEIVED") or _is_foul(typ, sub):
        return True
    if typ == "SHOT":
        # "GOAL KICK" is a restart marker, never a goal.
        goal = "GOAL" in sub.replace("GOAL KICK", "")
        return "OUT" in sub or goal
    return False


def labels_from_events(ev, n_frames, periods):
    """Per-frame int8 labels. ``periods`` is a list of (period, first_index, last_index)."""
    labels = np.full(n_frames, X, dtype=np.int8)
    team_code = {"H": H, "A": A}
    ev = ev.sort_values("start", kind="stable")
    for period, first, last in periods:
        p = ev[ev["period"] == period]
        starts = p["start"].to_numpy()
        ends = p["end"].to_numpy()
        teams = p["team"].tolist()
        types = p["type"].tolist()
        subs = p["subtype"].tolist()
        n = len(p)
        last = min(int(last), n_frames - 1)
        # Alive possession owner: each owner event sets the label to the end of the period.
        for i in range(n):
            if types[i] in OWNER_TYPES:
                labels[max(int(starts[i]), first):last + 1] = team_code[teams[i]]
        # Dead spells.
        for i in range(n):
            if not _is_trigger(types[i], subs[i]):
                continue
            ds = int(ends[i]) if ends[i] >= starts[i] else int(starts[i])
            de = last + 1  # dead to period end unless a later event stops it
            for j in range(i + 1, n):
                if starts[j] <= starts[i] or starts[j] < ds:
                    continue  # concurrent with the trigger itself (e.g. Game 3 logs a CARRY into the out)
                restart = _is_restart(types[j], subs[j])
                if restart or types[j] in CONTROL_TYPES:
                    if not restart and _is_foul(types[i], subs[i]):
                        de = None  # advantage play: no stoppage at all
                    else:
                        de = int(starts[j])
                    break
            if de is None:
                continue
            ds = max(ds, first)
            de = min(de, last + 1)
            if de > ds:
                labels[ds:de] = D
        # Before the first owner-setting event of the period nothing is known.
        owner_idx = [i for i in range(n) if types[i] in OWNER_TYPES]
        if owner_idx:
            labels[first:max(int(starts[owner_idx[0]]), first)] = X
        else:
            labels[first:last + 1] = X
    return labels


def _game3_meta():
    meta = SAMPLE_DIR / "Sample_Game_3" / "Sample_Game_3_metadata.xml"
    import xml.etree.ElementTree as ET

    root = ET.parse(str(meta)).getroot()
    home_id = next(e.get("id") for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "Team")
    _, _, params, _ = _parse_meta(meta)
    return meta, home_id, _periods_from_params(params)


def sample_game(n):
    """Return (Tracking, labels) for sample game ``n`` (1-3); tracking cached to cache/sample_{n}.npz."""
    d = SAMPLE_DIR / ("Sample_Game_%d" % n)
    if n == 3:
        meta, home_id, periods = _game3_meta()
        events = load_events_json(d / "Sample_Game_3_events.json", home_id)
    else:
        events = load_events_csv(d / ("Sample_Game_%d_RawEventsData.csv" % n))
        periods = None
    cache = CACHE_DIR / ("sample_%d.npz" % n)
    if cache.exists():
        z = np.load(cache)
        t = Tracking(z["players"], z["team"], z["ball"], [tuple(int(v) for v in r) for r in z["periods"]], PITCH)
    else:
        if n == 3:
            t = read_epts(d / "Sample_Game_3_tracking.txt", meta, PITCH, periods)
        else:
            t = read_metrica_csv(d / ("Sample_Game_%d_RawTrackingData_Home_Team.csv" % n),
                                 d / ("Sample_Game_%d_RawTrackingData_Away_Team.csv" % n), PITCH)
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(cache, players=t.players, team=t.team, ball=t.ball,
                 periods=np.asarray(t.periods, dtype=np.int64))
    return t, labels_from_events(events, len(t.ball), t.periods)
