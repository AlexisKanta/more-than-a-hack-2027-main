import json

import numpy as np
import pandas as pd

from bsd.events import labels_from_events, load_events_csv, load_events_json

H, A, D, X = 0, 1, 2, 3


def ev(rows):
    """rows: (team, type, subtype, start[, end])"""
    out = []
    for r in rows:
        team, typ, sub, start = r[:4]
        end = r[4] if len(r) > 4 else start
        out.append(dict(team=team, type=typ, subtype=sub, period=1, start=start, end=end))
    return pd.DataFrame(out).sort_values("start", kind="stable").reset_index(drop=True)


def lab(rows, n=120, periods=((1, 0, 119),)):
    return labels_from_events(ev(rows), n, list(periods))


def test_ball_out_until_throw_in():
    y = lab([("A", "PASS", "", 10), ("A", "BALL OUT", "", 20, 22), ("H", "SET PIECE", "THROW IN", 40)])
    assert (y[:10] == X).all()
    assert (y[10:22] == A).all()
    assert (y[22:40] == D).all()
    assert (y[40:] == H).all()


def test_recovery_switches_ball_lost_does_not():
    y = lab([("H", "PASS", "", 0), ("H", "BALL LOST", "INTERCEPTION", 10), ("A", "RECOVERY", "", 15)])
    assert (y[10:15] == H).all()
    assert (y[15:] == A).all()


def test_goal_dead_until_kickoff():
    y = lab([("H", "PASS", "", 10), ("H", "SHOT", "ON TARGET-GOAL", 50), ("A", "SET PIECE", "KICK OFF", 90)])
    assert (y[50:90] == D).all()
    assert y[90] == A and y[49] == H


def test_outside_periods_x():
    e = ev([("H", "PASS", "", 5), ("A", "PASS", "", 60)])
    e.loc[1, "period"] = 2
    y = labels_from_events(e, 100, [(1, 0, 49), (2, 55, 99)])
    assert (y[:5] == X).all() and (y[5:50] == H).all()
    assert (y[50:60] == X).all() and (y[60:] == A).all()


def test_before_first_event_of_period_x():
    y = lab([("H", "PASS", "", 30)])
    assert (y[:30] == X).all() and (y[30:] == H).all()


def test_json_subtype_lists(tmp_path):
    def e(i, sub, f):
        return {"team": {"id": "FIFATMA"}, "type": {"name": "set piece"}, "subtypes": sub,
                "start": {"frame": f}, "end": {"frame": f}, "period": 1}
    data = {"data": [e(1, [{"name": "FREE KICK"}, {"name": "INDIRECT"}], 11), e(2, None, 5),
                     e(3, {"name": "corner"}, 20)], "metadata": {}}
    p = tmp_path / "e.json"
    p.write_text(json.dumps(data))
    df = load_events_json(p, "FIFATMA")
    assert list(df.start) == [4, 10, 19]  # 1-based frames -> index
    assert df.subtype.tolist() == ["", "FREE KICK-INDIRECT", "CORNER"]
    assert df.team.tolist() == ["H", "H", "H"]
    assert df.type.iloc[0] == "SET PIECE"


def test_csv_loader(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
        "Away,PASS,,1,11,0.4,15,0.6,a,b,0,0,0,0\n"
        "Home,SET PIECE,KICK OFF,1,1,0.04,0,0,a,,NaN,NaN,NaN,NaN\n")
    df = load_events_csv(p)
    assert df.start.tolist() == [0, 10]
    assert df.team.tolist() == ["H", "A"]
    assert df.end.tolist() == [0, 14]  # End Frame 0 -> start
    assert df.subtype.tolist() == ["KICK OFF", ""]


def test_goal_kick_is_restart_even_as_ball_lost():  # R5
    y = lab([("H", "PASS", "", 10), ("H", "BALL OUT", "", 20), ("A", "BALL LOST", "GOAL KICK-INTERCEPTION", 50),
             ("H", "PASS", "", 60)])
    assert (y[20:50] == D).all() and y[50] == H  # BALL LOST does not switch owner
    assert y[60] == H


def test_goal_kick_pass_is_restart():
    y = lab([("H", "PASS", "", 10), ("H", "BALL OUT", "", 20), ("A", "PASS", "GOAL KICK", 50)])
    assert (y[20:50] == D).all() and y[50] == A


def test_foul_advantage_no_dead():  # R6
    y = lab([("H", "PASS", "", 10), ("A", "FAULT RECEIVED", "", 30), ("A", "PASS", "", 35)])
    assert (y[10:30] == H).all() and (y[30:35] == H).all() and (y[35:] == A).all()


def test_challenge_fault_advantage_no_dead():
    y = lab([("H", "PASS", "", 10), ("H", "CHALLENGE", "TACKLE-FAULT-LOST", 30), ("H", "RECOVERY", "", 33)])
    assert not (y == D).any()


def test_foul_then_free_kick_dead():
    y = lab([("H", "PASS", "", 10), ("A", "FAULT RECEIVED", "", 30), ("A", "SET PIECE", "FREE KICK", 70)])
    assert (y[30:70] == D).all() and y[70] == A


def test_ball_out_ends_at_control_event_without_restart():  # R6
    y = lab([("H", "PASS", "", 10), ("H", "BALL OUT", "", 20), ("A", "RECOVERY", "", 45)])
    assert (y[20:45] == D).all() and y[45] == A


def test_shot_out_ends_at_control_event():
    y = lab([("H", "SHOT", "OFF TARGET-OUT", 10), ("A", "PASS", "", 30)])
    assert (y[10:30] == D).all() and y[30] == A


def test_shot_not_out_is_not_trigger_and_goal_kick_not_goal():
    y = lab([("H", "SHOT", "ON TARGET-SAVED", 10), ("A", "RECOVERY", "", 12)])
    assert not (y == D).any()
    y = lab([("H", "SHOT", "BLOCKED", 10), ("H", "PASS", "GOAL KICK", 30)])
    assert not (y == D).any()


def test_trigger_without_restart_dead_to_period_end():
    y = lab([("H", "PASS", "", 10), ("H", "BALL OUT", "", 20)], n=60, periods=((1, 0, 49),))
    assert (y[20:50] == D).all() and (y[50:] == X).all()


def test_end_before_start_uses_start():
    y = lab([("H", "PASS", "", 10), ("H", "BALL OUT", "", 20, -1), ("A", "SET PIECE", "CORNER KICK", 40)])
    assert (y[20:40] == D).all()


def test_concurrent_carry_does_not_end_stoppage():
    # Game 3 style: a CARRY sharing the BALL OUT start frame (logged after it) is part of the same action.
    y = lab([("H", "PASS", "", 10), ("H", "BALL OUT", "", 20, 30), ("H", "CARRY", "", 20, 28),
             ("A", "SET PIECE", "THROW IN", 60)])
    assert (y[30:60] == D).all() and y[60] == A
