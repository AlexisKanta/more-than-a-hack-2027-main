"""Smart Tagging as fixed-weight evidence: ``tag_bonus`` is added to the (F, 3) log-scores.

Smart Tagging is automatic (not ground truth), so the weights are small, fixed, first-principles
nats and nothing here is tuned. Columns are the scorer's H=0, A=1, D=2. Codes and tag values are
matched case-insensitively. Rules:

* SET PIECES: +2.0 on D over the 3 s before the start (the ball is dead before a restart) and
  +1.0 on the tagged team over the first 2 s (nothing if ``Team`` is ``N/A``, e.g. throw-ins).
* GOALS: +2.0 on D from the goal's end to the next kick-off set piece (capped at 90 s).
* Any other team-tagged phase: +0.3 on that team over its interval.
* R14: inside each period, frames before the first "kick off" set piece get +3.0 on D.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

H, A, D = 0, 1, 2
HOME_NAME = "barcelona"  # the only club-name mapping: any other named team is the away side
SET_PIECE_DEAD_S, SET_PIECE_DEAD_NATS = 3.0, 2.0
SET_PIECE_TEAM_S, SET_PIECE_TEAM_NATS = 2.0, 1.0
GOAL_DEAD_NATS, GOAL_DEAD_CAP_S = 2.0, 90.0
PHASE_TEAM_NATS = 0.3
PRE_KICKOFF_DEAD_NATS = 3.0


def _team_col(tags, home_name=HOME_NAME):
    team = str(tags.get("Team") or "").strip().lower()
    if not team or team == "n/a":
        return None
    return H if home_name in team else A


def _is_kickoff(ev):
    return ev["code"].lower() == "set pieces" and "kick off" in str(ev["tags"].get("Type", "")).lower()


def bonus_from_events(events, n_frames, periods, fps=25):
    """Pure function: Smart Tagging event dicts (starter loader format) -> (F, 3) bonus."""
    out = np.zeros((n_frames, 3))

    def add(a, b, col, nats):
        a, b = max(int(a), 0), min(int(b), n_frames)
        if b > a:
            out[a:b, col] += nats

    kickoffs = sorted(e["start_frame"] for e in events if _is_kickoff(e))
    for ev in events:
        code = ev["code"].lower()
        col = _team_col(ev["tags"])
        s, e = ev["start_frame"], ev["end_frame"]
        if code == "set pieces":
            add(s - round(SET_PIECE_DEAD_S * fps), s, D, SET_PIECE_DEAD_NATS)
            if col is not None:
                add(s, s + round(SET_PIECE_TEAM_S * fps), col, SET_PIECE_TEAM_NATS)
        elif code == "goals":
            cap = e + round(GOAL_DEAD_CAP_S * fps)
            nxt = next((k for k in kickoffs if k >= e), cap)
            add(e, min(nxt, cap), D, GOAL_DEAD_NATS)
        elif col is not None:
            add(s, e, col, PHASE_TEAM_NATS)
    for _, first, last in periods:
        k = next((k for k in kickoffs if first <= k <= last), None)
        if k is not None:
            add(first, k, D, PRE_KICKOFF_DEAD_NATS)
    return out


def tag_bonus(n_frames, fps=25, periods=None, events=None):
    """Bonus for BAR-ATH; loads the tagging file and the graded blocks unless given."""
    if events is None:
        sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
        import load_smart_tagging

        events = load_smart_tagging.load(fps=fps)
    if periods is None:
        import games

        periods = games.half_video_blocks("BAR-ATH")
    return bonus_from_events(events, n_frames, periods, fps)
