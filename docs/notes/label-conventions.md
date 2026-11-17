# Label conventions: public GT window vs. our sample-game event labels

Task 8, Step 1. A qualitative check of what the organisers' ground truth means by "dead" and by
"owning", compared with the rules that turn the Metrica sample-game event logs into training and
bench labels (`bsd/events.py`). Nothing here was fitted: no threshold, cost, weight or window was
chosen from the public window, and our submission was not scored against it. Frame numbers below
are examples for the reader only; none of them appears in `bsd/`.

## What was looked at

- `evaluation/gt/BAR-ATH_possession_gt_public.csv`, video frames 11185-18684 (5 min, 21 segments:
  17 possessions, 4 dead spells).
- BAR-ATH tracking features from `bsd.features.frame_features` around every GT boundary:
  `ball_visible`, `ball_out`, `ball_dist_edge`, `ball_speed`, `near_home` / `near_away`,
  `speed_all`, `ball_static`.
- Smart Tagging intervals overlapping the window (`starter/load_smart_tagging.py`).
- No video (subagents cannot watch it, ruling).
- For comparison: the event logs and clean tracking of sample games 1-3.

## GT segments in the window

| label | frames | s | context |
|---|---|---|---|
| A | 11185-11351 | 6.6 | kick-off; kicking team owns from the first graded frame, no D at kick-off |
| H/A/H | 11351-12665 | 8.1, 2.8, 41.6 | direct hand-overs, no D between |
| A | 12665-12851 | 7.4 | |
| D | 12851-12989 | 5.5 | ball lost by the tracker ~0.5 s before D starts; restart by H |
| H/A/H | 12989-14846 | 28.6, 3.2, 42.4 | direct hand-overs |
| A/H/A/H/A | 14846-15889 | 20.0, 0.8, 16.0, 1.3, 3.6 | two very short H spells inside duels |
| D | 15889-16400 | 20.4 | ball stays tracked inside the pitch; non-out stoppage (whistle) |
| A, H | 16400-17380 | 35.7, 3.5 | |
| D | 17380-17838 | 18.3 | Smart Tagging: free kick, Athletic; ball tracked inside, moving |
| A | 17838-18027 | 7.6 | |
| D | 18027-18389 | 14.5 | ball crosses the line; Smart Tagging: goal kick, Barcelona |
| H | 18389-18684 | 11.8 | (window ends) |

## Conventions observed

**When a dead spell starts.**
- Ball out of play: D starts at the line crossing. At 18027 the tracked ball's `ball_out` flag
  turns on one frame before the first D frame.
- Fouls and other whistles: D starts while the ball is still inside the pitch and often still
  moving (at 17380 the ball is travelling at 7-12 m/s; at 15889 it is rolling at ~7 m/s and
  slowing, nobody within 7 m). That is the whistle, not a line crossing and not "the moment the
  ball comes to rest".
- Collective pace drops *after* the D start, over the next 1-3 s (`speed_all` ~2.5 -> ~1.2 m/s
  in 12851-12880, ~2.2 -> ~1.2 m/s in 15896-15956). Walking pace is a consequence of a stoppage,
  not its start.

**When a dead spell ends.** At the restart touch/kick, i.e. when the restarting player moves a
placed ball, not when the ball is placed. At 15889-16400 an Athletic player stands at a resting
ball (`near_away` 0.07 m, ball speed ~0.2 m/s) for ~2 s before the label turns A as the ball
starts moving. At 12989 and 18389 the same: a home player is within 0.3 m of the ball for a while
inside D, and H starts when the ball is played. During a stoppage the tracked ball can be moved
around (picked up, rolled to the taker) with players of either team next to it, still D.

**Who owns after a restart.** The restarting team, from the restart frame (A after the A free
kick, H after the H goal kick and after the 12989 restart). There is no "neutral" stretch after a
restart.

**Possession changes without a dead spell.** Most changes are direct (12 of the 16 changes
between consecutive possessions in the window go H<->A without D in between; samples: ~73 %). In the tracking, the GT switch
sits at the first touch of the new team -- e.g. 12665 switches to A while a pass is in flight,
right after an Athletic player came within ~2 m of the ball and before the ball reaches the next
Athletic player. The losing team's last pass does not hand the ball over; the new team's first
control does.

**Contested touches.** Brief controls by the other team *do* count as possessions: 15346-15365
(0.8 s) and 15766-15799 (1.3 s) are home possessions inside duels where both teams have a player
within ~1-3 m of the ball. On BAR-ATH's tracking the nearest-team cue does not show these clearly
(the away player is often the nearer one throughout), so they are hard to detect from tracking,
but the GT convention is that a short control is still a possession. Whether a mere deflection
(no control) switches cannot be told without video; nothing in the window contradicts "control
switches, a touch without control does not".

**Smart Tagging** intervals are phases, not D spells: the set-piece tags overlap the D spell and
the following possession and start/end seconds away from the GT boundaries (free kick 17533-18408
vs D 17380-17838; goal kick 18320-19195 vs D 18027-18389). Their Team tag is the restarting team,
i.e. the team that owns after the D spell. One phase (attacking transition, Athletic) ends
exactly where the 12851 D spell starts.

## Comparison with the event-label rules (`bsd/events.py`)

| convention in GT | our rule | verdict |
|---|---|---|
| D starts at the line crossing | D starts at the `BALL OUT` event's end frame; in all three sample games that frame is the first frame the clean ball is beyond the line (median offset 0 frames, 10th-90th pct 0) | consistent |
| D starts at the whistle for fouls, ball may still move | D starts at `FAULT RECEIVED` / foul `CHALLENGE` | consistent |
| D ends at the restart touch | D ends at the start of the next `SET PIECE` (or goal-kick-tagged event), else at the next control event | consistent |
| restarting team owns from the restart | `SET PIECE` is an owner event | consistent |
| new team owns from its first control, not from the lost pass | `RECOVERY` / `PASS` / `CARRY` / `SHOT` switch; `BALL LOST` does not | consistent |
| short controls (~1 s) by the other team are possessions | any `RECOVERY` etc. switches, however short (sample labels: 19-23 % of possessions under 2 s; window: 2 of 17) | consistent |
| kick-off frame is alive and owned by the kicking team | `SET PIECE` `KICK OFF` sets the owner at its start; before it X (unscored) | consistent for scoring (GT labels pre-kick-off frames D, e.g. the 2nd-half block before kick-off; our sample periods start at kick-off, and X frames are not scored by the bench) |

**No rule was changed.** None of the GT conventions contradicts a rule in `bsd/events.py`.

## The dead-share / possession-length difference

Sample labels have D share 28-40 % and mean possession ~10-12 s; the window has 19.6 % D and a
14.2 s mean over 17 possessions. This is not a definition difference:

- The window's D spells (5.5, 20.4, 18.3, 14.5 s) all fall inside the sample games' D-spell
  10th-90th percentile range (roughly 4-50 s, median 11-18 s). Individual stoppages are not
  shorter in the GT.
- The window simply contains few stoppages (4 in 5 min, ~0.8 per minute vs ~1 per minute in the
  samples) and several long possessions (41.6, 42.4, 35.7 s). Its median possession (7.6 s) is
  close to the samples' (5.1-6.5 s); the mean is driven by three long spells.
- Five minutes with 17 possessions and 4 stoppages is far too small a sample to read a match-level
  dead share or mean possession from. Ball-in-play time in professional football is typically
  around 55-65 % of a match, i.e. D ~35-45 %, close to the samples' 28-40 % and well above the window's 20 %.
