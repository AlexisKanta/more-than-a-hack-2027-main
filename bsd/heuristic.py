"""Hand-set per-frame H/A/D evidence from ``frame_features`` (Milestone 1, no learning).

``heuristic_logp(X, names)`` returns an (F, 3) array of log-scores in nats, columns H, A, D.
Only differences between columns matter to the decoder. The score is built from two numbers:

    team  (> 0 favours home, < 0 away):  logp[H] = +team / 2, logp[A] = -team / 2
    dead  (> 0 favours a stoppage):      logp[D] = dead

Every term is bounded so that one frame contributes at most about 3 nats; the decoder's
transition costs (``HEURISTIC_PARAMS``), not single frames, decide where labels change. A term
whose inputs are undefined (NaN, e.g. ball columns while the ball is not tracked) contributes 0,
so every row is finite.

Team evidence
-------------
* Ball visible -- proximity (``_near_team``): the team whose nearest player is closer to the
  ball is the likelier owner. The gap ``near_away - near_home`` is clipped to +-4 m (beyond that
  the further team is simply "not involved") and weighted by ``exp(-min(near) / 2 m)``: a player
  within 1-2 m of the ball can play it (control), while when nobody is within ~4 m the ball is in
  flight or loose and proximity says little. Peak contribution 0.5 * 4 = 2 nats.
* Ball visible -- crowding (``_n2_team``): each extra player of one team within 2 m adds 0.5 nats
  (clipped to +-1). Close support around the ball is typical of the team keeping it, and it
  breaks ties when both nearest players are close (a duel).
* Fast ball (``ball_speed`` > ~8 m/s, a pass or shot) scales both ball terms down by
  ``1 / (1 + (speed / 8)^2)``: during a pass the nearest player is usually an opponent the ball
  is passing, not the owner.
* Context, always (``_context_team``): the attacking team sits higher up the pitch and moves
  forward, so ``depth_diff_m3s`` (home minus away centroid depth in each team's attacking frame)
  and ``vx_fwd_home_m3s - vx_fwd_away_m3s`` lean towards the attacker. These describe where the
  play is rather than who has the ball, so each is capped at 0.3 nats (tanh of 30 m / 2 m/s):
  enough to carry a possession through the ~40 % of frames without a tracked ball, too weak to
  overrule the ball terms.

Dead-ball evidence
------------------
* ``ball_out`` (ball tracked beyond the lines): +3, the one direct stoppage signal.
* Collective pace (``speed_all_m1s``): in live play players jog or run (median ~1.5 m/s); during
  stoppages they walk (<~1 m/s). ``1.5 * clip((1.0 - s) / 0.5, -1, 1)`` gives +1.5 at 0.5 m/s
  and -1.5 (alive) at 1.5 m/s and above.
* Static ball (``ball_static``): a ball resting for 1 s is being placed for a restart or held:
  +0.5 anywhere (free kick, kick-off), +1.0 more within 2 m of a line or beyond it (throw-in,
  corner, goal kick).
* Moving ball (``ball_speed_m1s`` > 5 m/s): -1, a ball travelling at passing speed is in play.

The dead score is clipped to [-2, 3].
"""
import numpy as np

from bsd.decode import A, D, H, DecodeParams

NEAR_CLIP = 4.0  # m
CONTROL_SCALE = 2.0  # m, e-folding distance of "can play the ball"
NEAR_WEIGHT = 0.5  # nats per metre of proximity gap
N2_WEIGHT, N2_CLIP = 0.5, 1.0
PASS_SPEED = 8.0  # m/s
DEPTH_SCALE, VX_SCALE, CONTEXT_WEIGHT = 30.0, 2.0, 0.3
OUT_WEIGHT = 3.0
WALK_SPEED, WALK_SPAN, WALK_WEIGHT = 1.0, 0.5, 1.5
STATIC_WEIGHT, STATIC_EDGE_WEIGHT, EDGE_DIST = 0.5, 1.0, 2.0
MOVING_BALL_SPEED, MOVING_BALL_WEIGHT = 5.0, 1.0
DEAD_CLIP = (-2.0, 3.0)

# Viterbi costs in nats. "Moderate" evidence is ~0.5 nats per frame (e.g. nearest players 1 m
# vs 3 m from the ball). A possession change must be backed by ~2 s of it: 2 * 25 * 0.5 = 25.
# Entering or leaving a stoppage costs 1 s of moderate evidence each, so a whole inserted dead
# spell (in + out) also needs ~2 s of support -- shorter "stoppages" are just slow play.
HEURISTIC_PARAMS = DecodeParams(switch_ha=25.0, alive_to_dead=12.5, dead_to_alive=12.5)


def _col(X, names, name):
    return X[:, names.index(name)].astype(np.float64)


def _z(v):
    """Undefined evidence counts as no evidence."""
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def heuristic_logp(X, names):
    """(F, K) features -> (F, 3) finite log-scores for H, A, D (see module docstring)."""
    names = list(names)
    c = lambda n: _col(X, names, n)  # noqa: E731
    visible = _z(c("ball_visible")) > 0

    near_h, near_a = c("near_home"), c("near_away")
    gap = np.clip(near_a - near_h, -NEAR_CLIP, NEAR_CLIP)
    control = np.exp(-np.maximum(np.fmin(near_h, near_a), 0.0) / CONTROL_SCALE)
    near_team = NEAR_WEIGHT * gap * control
    n2_team = np.clip(N2_WEIGHT * (c("n2_home") - c("n2_away")), -N2_CLIP, N2_CLIP)
    damp = 1.0 / (1.0 + (_z(c("ball_speed")) / PASS_SPEED) ** 2)
    ball_team = np.where(visible, damp * (_z(near_team) + _z(n2_team)), 0.0)

    vx_diff = c("vx_fwd_home_m3s") - c("vx_fwd_away_m3s")
    context_team = CONTEXT_WEIGHT * (_z(np.tanh(c("depth_diff_m3s") / DEPTH_SCALE))
                                     + _z(np.tanh(vx_diff / VX_SCALE)))
    team = ball_team + context_team

    out = OUT_WEIGHT * (_z(c("ball_out")) > 0.5)
    walk = _z(WALK_WEIGHT * np.clip((WALK_SPEED - c("speed_all_m1s")) / WALK_SPAN, -1, 1))
    static = _z(c("ball_static")) > 0
    near_edge = c("ball_dist_edge") < EDGE_DIST  # NaN (ball not tracked) compares False
    placed = np.where(static, STATIC_WEIGHT + STATIC_EDGE_WEIGHT * near_edge, 0.0)
    moving = -MOVING_BALL_WEIGHT * (_z(c("ball_speed_m1s")) > MOVING_BALL_SPEED)
    dead = np.clip(out + walk + placed + moving, *DEAD_CLIP)

    logp = np.zeros((len(X), 3))
    logp[:, H] = team / 2
    logp[:, A] = -team / 2
    logp[:, D] = dead
    return logp
