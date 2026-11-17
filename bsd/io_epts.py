"""Streaming parser for FIFA/EPTS raw tracking (``frame:x,y[,s];...:ball``) + metadata XML."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from bsd.tracking import Tracking

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "bar_ath.npz"


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _iter(root, name):
    return (e for e in root.iter() if _local(e.tag) == name)


def _parse_meta(meta_path):
    """Return (team_of_slot, specs, period_params, n_slots).

    ``specs`` is a list of (start_frame, end_frame, [slot per group]) taken from the
    DataFormatSpecifications (squads may change over the match); empty if absent.
    """
    root = ET.parse(str(meta_path)).getroot()
    team_ids = [e.get("id") for e in _iter(root, "Team")]
    player_team = {e.get("id"): e.get("teamId") for e in _iter(root, "Player")}
    chan_player, slot_of = {}, {}
    for e in _iter(root, "PlayerChannel"):
        chan_player[e.get("id")] = e.get("playerId")
        slot_of.setdefault(e.get("playerId"), len(slot_of))
    team = np.zeros(len(slot_of), dtype=np.int8)
    for pid, s in slot_of.items():
        team[s] = 0 if player_team[pid] == team_ids[0] else 1

    specs = []
    for spec in _iter(root, "DataFormatSpecification"):
        groups = []
        for sr in spec:
            if _local(sr.tag) != "SplitRegister":
                continue
            for g in sr:  # one child SplitRegister per player group
                refs = [r.get("playerChannelId") for r in g if _local(r.tag) == "PlayerChannelRef"]
                if refs:
                    groups.append(slot_of[chan_player[refs[0]]])
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), groups))

    params = {}
    for p in _iter(root, "ProviderParameter"):
        name, val = None, None
        for c in p:
            if _local(c.tag) == "Name":
                name = (c.text or "").strip()
            elif _local(c.tag) == "Value":
                val = (c.text or "").strip()
        if name and val:
            params[name] = val
    return team, specs, params, len(slot_of)


def _periods_from_params(params):
    out = []
    for n, half in ((1, "first_half"), (2, "second_half")):
        s, e = params.get(half + "_start"), params.get(half + "_end")
        if s and e:
            out.append((n, int(s) - 1, int(e) - 1))
    return out


def read_epts(raw_path, meta_path, pitch, periods=None):
    team, specs, params, n_slots = _parse_meta(meta_path)
    if periods is None:
        periods = _periods_from_params(params)
    with open(raw_path, "r") as fh:
        n_frames = sum(1 for _ in fh)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    spec_slots = [np.asarray(g, dtype=np.intp) for _, _, g in specs]

    def slots_for(frame_id, n_groups):
        for (s, e, _), sl in zip(specs, spec_slots):
            if s <= frame_id <= e and len(sl) == n_groups:
                return sl
        if specs:
            raise ValueError(
                "no DataFormatSpecification matches frame %d with %d groups" % (frame_id, n_groups)
            )
        return np.arange(n_groups, dtype=np.intp)  # no spec: file order == slot order

    with open(raw_path, "r") as fh:
        for i, line in enumerate(fh):
            parts = line.strip().split(":")
            if len(parts) < 2:
                continue
            if int(parts[0]) != i + 1:
                raise ValueError("frame id %s at line %d breaks the 1-based sequence" % (parts[0], i + 1))
            groups = parts[1].split(";")
            toks = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            w = len(toks) // len(groups)
            xy = toks.reshape(len(groups), w)[:, :2]
            players[i, slots_for(int(parts[0]), len(groups))] = xy
            if len(parts) > 2 and parts[2]:
                b = parts[2].split(",")
                if len(b) >= 2:
                    ball[i] = np.array(b[:2], dtype=np.float32)
    scale = np.array(pitch, dtype=np.float32)
    players *= scale
    ball *= scale
    return Tracking(players, team, ball, list(periods), (float(pitch[0]), float(pitch[1])))


def bar_ath_tracking():
    import games

    pitch = games.GAMES["BAR-ATH"]["pitch"]
    periods = [(int(p), int(a), int(b)) for p, a, b in games.half_video_blocks("BAR-ATH")]
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        return Tracking(z["players"], z["team"], z["ball"], periods, (float(pitch[0]), float(pitch[1])))
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"])
    t = read_epts(raw_path, meta_path, pitch, periods)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez(CACHE_PATH, players=t.players, team=t.team, ball=t.ball)
    return t
