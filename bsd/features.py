"""Per-frame, slot-agnostic team features computed from a ``Tracking``.

Every column is a function of the frame (and, for windowed columns, of frames in the same
period). Names follow ``<base>[_home|_away|_diff][_m1s|_m3s]``; ``mirror`` relies on that to
swap the teams. Rows outside ``tr.periods`` are NaN, and windows never cross a period boundary.

Team-relative quantities use the team's attacking direction from ``attack_sign``: ``cx``/``cy``
are the team centroid in that team's own frame (pitch rotated so it attacks +x; 0 = own goal
line), ``vx_fwd`` is its mean x-velocity towards the goal it attacks, and ``ball_xfwd_<team>``
is the ball x in that frame. ``ball_ywide`` = |ball y - W/2| is team-neutral.
"""
import numpy as np

MAX_SLOT_SPEED = 12.0  # m/s; a faster implied step is a tracklet identity jump, not motion
OUT_MARGIN = 0.5  # m beyond the lines before the ball counts as out
NEAR_RADII = (2.0, 5.0)  # m, player counts around the ball
BALL_SPEED_SPAN = 2  # frames each side of the centred ball-speed difference
STATIC_SPEED = 1.0  # m/s
STATIC_S = 1  # ball must stay under STATIC_SPEED for this long
TREND_S = 2  # speed_all_trend compares the last TREND_S seconds with the TREND_S before
FSB_CAP_S = 30  # frames_since_ball cap, seconds
TURN_SPAN = 5  # frames each side for the ball-direction change (hand-over features)
FLIP_SPAN = 6  # frames each side for the nearest-team flip
CLOSE_R = 2.0  # m, closest player counts as holding the ball within this radius
HANDOVER_CTX = ("ball_acc", "ball_turn", "near_flip", "closest_home", "closest_away")
CONTEXT_S = {"m1s": 1, "m3s": 3}  # centred rolling-mean half-widths, seconds
CONTEXT_COLS = (
    "near_home", "near_away", "n5_home", "n5_away", "ball_out", "ball_speed",
    "depth_diff", "vx_fwd_home", "vx_fwd_away", "speed_all",
)


def _segments(tr):
    """Half-open ``(start, stop)`` frame ranges of the periods."""
    return [(int(a), int(b) + 1) for _, a, b in tr.periods]


def _period_id(tr, n):
    pid = np.full(n, -1, dtype=np.int64)
    for k, (a, b) in enumerate(_segments(tr)):
        pid[a:b] = k
    return pid


def _ratio(num, den):
    out = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _slot_mean(vals, sel):
    """NaN-aware mean over the slots selected by boolean ``sel``; NaN when none is finite."""
    v = vals[:, sel]
    ok = np.isfinite(v)
    return _ratio(np.where(ok, v, 0).sum(axis=1, dtype=np.float64), ok.sum(axis=1))


def _window(x, segs, lo, hi):
    """(sum, count) of finite ``x`` over ``[t+lo, t+hi]`` clipped to t's period segment."""
    s = np.zeros(len(x), dtype=np.float64)
    c = np.zeros(len(x), dtype=np.int64)
    for a, b in segs:
        seg = x[a:b]
        ok = np.isfinite(seg)
        cs = np.concatenate(([0.0], np.cumsum(np.where(ok, seg, 0), dtype=np.float64)))
        cn = np.concatenate(([0], np.cumsum(ok)))
        t = np.arange(b - a)
        i0 = np.clip(t + lo, 0, b - a)
        i1 = np.clip(t + hi + 1, 0, b - a)
        s[a:b] = cs[i1] - cs[i0]
        c[a:b] = cn[i1] - cn[i0]
    return s, c


def _window_mean(x, segs, lo, hi):
    return _ratio(*_window(x, segs, lo, hi))


def attack_sign(tr):
    """(F,) int8: +1 if home attacks +x in the frame's period, -1 if -x, 0 outside periods.

    Per period, home attacks +x when its mean centroid x lies left of the away team's.
    """
    home = tr.team == 0
    x = tr.players[..., 0]
    out = np.zeros(len(tr.ball), dtype=np.int8)
    for a, b in _segments(tr):
        ch = _slot_mean(x[a:b], home)
        ca = _slot_mean(x[a:b], ~home)
        mh = ch[np.isfinite(ch)].mean() if np.isfinite(ch).any() else np.nan
        ma = ca[np.isfinite(ca)].mean() if np.isfinite(ca).any() else np.nan
        out[a:b] = -1 if mh > ma else 1  # missing data -> +1 (team-relative columns are NaN then)
    return out


def _slot_velocity(tr, pid):
    """(F,S,2) per-slot velocity in m/s from consecutive frames of the same period."""
    P = tr.players
    V = np.full(P.shape, np.nan, dtype=np.float32)
    V[1:] = (P[1:] - P[:-1]) * np.float32(tr.fps)
    V[1:][pid[1:] != pid[:-1]] = np.nan
    V[np.hypot(V[..., 0], V[..., 1]) > MAX_SLOT_SPEED] = np.nan
    return V


def _handover_features(f, bx, by, vis, pid, segs, n, fps):
    """Hand-over columns, added to ``f`` in place (see ``frame_features(handover=True)``)."""
    k, S = BALL_SPEED_SPAN, TURN_SPAN
    # ball acceleration: centred difference of the centred ball velocity
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    if n > 2 * k:
        same = (pid[2 * k:] == pid[:-2 * k]) & (pid[2 * k:] >= 0)
        vx[k:n - k] = np.where(same, (bx[2 * k:] - bx[:-2 * k]) * fps / (2 * k), np.nan)
        vy[k:n - k] = np.where(same, (by[2 * k:] - by[:-2 * k]) * fps / (2 * k), np.nan)
    acc = np.full(n, np.nan)
    if n > 2 * k:
        acc[k:n - k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k]) * fps / (2 * k)
    f["ball_acc"] = acc
    # ball turn: angle between the previous and next S-frame displacement
    turn = np.full(n, np.nan)
    if n > 2 * S:
        c = slice(S, n - S)
        px, py = bx[c] - bx[:n - 2 * S], by[c] - by[:n - 2 * S]
        qx, qy = bx[2 * S:] - bx[c], by[2 * S:] - by[c]
        same = (pid[2 * S:] == pid[:-2 * S]) & (pid[2 * S:] >= 0)
        ok = same & (np.hypot(px, py) > 0) & (np.hypot(qx, qy) > 0)
        ang = np.abs(np.arctan2(px * qy - py * qx, px * qx + py * qy))
        turn[c] = np.where(ok, ang, np.nan)
    f["ball_turn"] = turn
    nd = f["near_home"] - f["near_away"]
    f["near_diff"] = nd
    flip = np.full(n, np.nan)
    F = FLIP_SPAN
    if n > 2 * F:
        before, after = nd[:n - 2 * F], nd[2 * F:]
        same = (pid[2 * F:] == pid[:-2 * F]) & (pid[2 * F:] >= 0)
        ok = same & np.isfinite(before) & np.isfinite(after)
        flip[F:n - F] = np.where(ok, (np.sign(before) != np.sign(after)).astype(np.float64), np.nan)
    f["near_flip"] = flip
    nh = np.where(np.isfinite(f["near_home"]), f["near_home"], np.inf)
    na = np.where(np.isfinite(f["near_away"]), f["near_away"], np.inf)
    f["closest_home"] = np.where(vis, ((nh < na) & (nh <= CLOSE_R)).astype(np.float64), np.nan)
    f["closest_away"] = np.where(vis, ((na < nh) & (na <= CLOSE_R)).astype(np.float64), np.nan)
    for name in HANDOVER_CTX:
        f[name + "_m1s"] = _window_mean(f[name], segs, -fps, fps)


def frame_features(tr, handover=False):
    """Return ``(X, names)``: X is (F, K) float32, NaN where a feature is undefined.

    ``handover=True`` appends the hand-over columns after the default ones.
    """
    n, fps = len(tr.ball), int(tr.fps)
    L, W = float(tr.pitch[0]), float(tr.pitch[1])
    segs = _segments(tr)
    pid = _period_id(tr, n)
    P, B = tr.players, tr.ball
    sign = attack_sign(tr).astype(np.float64)
    teams = {"home": tr.team == 0, "away": tr.team == 1}
    direction = {"home": sign, "away": -sign}
    f = {}

    # --- ball -------------------------------------------------------------------------
    vis = np.isfinite(B).all(axis=1)
    bx, by = B[:, 0].astype(np.float64), B[:, 1].astype(np.float64)
    f["ball_visible"] = vis.astype(np.float64)
    out = (bx < -OUT_MARGIN) | (bx > L + OUT_MARGIN) | (by < -OUT_MARGIN) | (by > W + OUT_MARGIN)
    f["ball_out"] = np.where(vis, out, np.nan)
    k = BALL_SPEED_SPAN
    bs = np.full(n, np.nan)
    if n > 2 * k:
        same = (pid[2 * k:] == pid[:-2 * k]) & (pid[2 * k:] >= 0)
        step = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) * fps / (2 * k)
        bs[k:n - k] = np.where(same, step, np.nan)
    f["ball_speed"] = bs
    f["ball_dist_edge"] = np.minimum(np.minimum(bx, L - bx), np.minimum(by, W - by))
    f["ball_xfwd_home"] = L / 2 + (bx - L / 2) * sign  # ball x in home's attacking frame
    f["ball_xfwd_away"] = L - f["ball_xfwd_home"]
    f["ball_ywide"] = np.abs(by - W / 2)
    d = np.hypot(P[..., 0] - B[:, None, 0], P[..., 1] - B[:, None, 1])  # (F,S), NaN if either missing
    for t, sel in teams.items():
        dt = d[:, sel]
        nearest = np.where(np.isfinite(dt), dt, np.inf).min(axis=1) if sel.any() else np.full(n, np.inf)
        f["near_" + t] = np.where(np.isfinite(nearest), nearest, np.nan)
        for r in NEAR_RADII:
            f["n%d_%s" % (r, t)] = np.where(vis, (dt < r).sum(axis=1), np.nan)
    fsb = np.full(n, np.nan)
    for a, b in segs:
        idx = np.arange(a, b)
        last = np.maximum.accumulate(np.where(vis[a:b], idx, -1))
        fsb[a:b] = np.where(last >= 0, np.minimum(idx - last, FSB_CAP_S * fps), np.nan)
    f["frames_since_ball"] = fsb

    # --- collective shape and motion -----------------------------------------------------
    V = _slot_velocity(tr, pid)
    spd = np.hypot(V[..., 0], V[..., 1])
    valid = np.isfinite(P).all(axis=2)
    for t, sel in teams.items():
        cxr, cyr = _slot_mean(P[..., 0], sel), _slot_mean(P[..., 1], sel)
        f["cx_" + t] = L / 2 + (cxr - L / 2) * direction[t]
        f["cy_" + t] = W / 2 + (cyr - W / 2) * direction[t]
        sq = (P[..., 0] - cxr[:, None].astype(np.float32)) ** 2 + (P[..., 1] - cyr[:, None].astype(np.float32)) ** 2
        f["spread_" + t] = np.sqrt(_slot_mean(sq, sel))
        f["speed_" + t] = _slot_mean(spd, sel)
        f["vx_fwd_" + t] = _slot_mean(V[..., 0], sel) * direction[t]
    f["depth_diff"] = f["cx_home"] - f["cx_away"]  # (cx - L/2) * direction, home minus away
    for t, sel in teams.items():
        f["n_tracked_" + t] = valid[:, sel].sum(axis=1).astype(np.float64)

    # --- stoppage cues ---------------------------------------------------------------------
    sa = _slot_mean(spd, np.ones(len(tr.team), dtype=bool))
    f["speed_all"] = sa
    m = TREND_S * fps
    f["speed_all_trend"] = _window_mean(sa, segs, -m + 1, 0) - _window_mean(sa, segs, -2 * m + 1, -m)
    w = STATIC_S * fps
    n_fast, n_seen = _window(np.where(np.isfinite(bs), bs >= STATIC_SPEED, np.nan), segs, -w + 1, 0)
    _, n_win = _window(np.ones(n), segs, -w + 1, 0)
    f["ball_static"] = np.where(n_fast > 0, 0.0, np.where((n_seen == n_win) & (n_win > 0), 1.0, np.nan))

    # --- temporal context ------------------------------------------------------------------
    for name in CONTEXT_COLS:
        for suffix, sec in CONTEXT_S.items():
            f["%s_%s" % (name, suffix)] = _window_mean(f[name], segs, -sec * fps, sec * fps)

    if handover:
        _handover_features(f, bx, by, vis, pid, segs, n, fps)

    names = list(f)
    X = np.empty((n, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = f[name]
    X[pid < 0] = np.nan
    return X, names


def _partner(name):
    if "_home" in name:
        return name.replace("_home", "_away")
    if "_away" in name:
        return name.replace("_away", "_home")
    return name


def mirror(X, names):
    """Swap home/away: exchange every ``_home``/``_away`` column pair, negate ``_diff`` columns."""
    idx = {nm: j for j, nm in enumerate(names)}
    missing = [nm for nm in names if _partner(nm) not in idx]
    if missing:
        raise ValueError("columns without a home/away partner: %s" % missing)
    perm = [idx[_partner(nm)] for nm in names]
    sign = np.array([-1 if "_diff" in nm else 1 for nm in names], dtype=X.dtype)
    return X[:, perm] * sign
