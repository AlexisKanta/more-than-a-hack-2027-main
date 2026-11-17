import numpy as np
import pandas as pd
import pytest

from bsd import experiments as ex


def _rows(spell, macro):
    keys = [(g, s) for s in (0, 1) for g in (1, 2, 3)]
    return pd.DataFrame({"game": [g for g, _ in keys], "seed": [s for _, s in keys],
                         "spell_f1": spell, "macro_f1": macro})


INC_SPELL = np.array([0.50, 0.52, 0.54, 0.56, 0.58, 0.60])
INC_MACRO = np.full(6, 0.80)


def _cand(deltas, d_macro):
    return _rows(INC_SPELL + np.array(deltas), INC_MACRO + d_macro)


def test_gate_rule():
    inc = _rows(INC_SPELL, INC_MACRO)
    # +0.012 mean, 4 of 6 better, macro -0.005 -> passes
    r = ex.gate(inc, _cand([0.03, 0.03, 0.01, 0.01, -0.003, -0.005], -0.005))
    assert r["n_better"] == 4 and r["d_spell"] == pytest.approx(0.012)
    assert r["d_macro"] == pytest.approx(-0.005)
    assert r["passed"] is True
    # +0.012 mean, only 3 of 6 better -> fails
    r = ex.gate(inc, _cand([0.03, 0.03, 0.02, -0.002, -0.003, -0.003], -0.005))
    assert r["n_better"] == 3 and r["d_spell"] == pytest.approx(0.012) and r["passed"] is False
    # +0.009 mean (all better) -> fails
    r = ex.gate(inc, _cand([0.009] * 6, 0.0))
    assert r["n_better"] == 6 and r["passed"] is False
    # +0.012 mean, 6 of 6 better, macro -0.011 -> fails
    r = ex.gate(inc, _cand([0.012] * 6, -0.011))
    assert r["passed"] is False


def test_gate_matches_cases_on_game_and_seed():
    inc = _rows(INC_SPELL, INC_MACRO)
    cand = _cand([0.012] * 6, 0.0).iloc[::-1].reset_index(drop=True)
    r = ex.gate(inc, cand)
    assert r["n_better"] == 6 and r["passed"] is True
    with pytest.raises(ValueError):
        ex.gate(inc, cand.iloc[:5])


def _fake_scorer(peak, calls):
    """score_points stub: spellF1 peaks at ``peak`` (a dict over the point's parameters)."""
    def score_points(points):
        calls.extend(points)
        rows = []
        for pt in points:
            v = -sum(abs(np.log2((pt[k] + 1) / (peak[k] + 1))) for k in peak)
            for g in (1, 2, 3):
                rows.append({**pt, "game": g, "seed": 0, "spell_f1": v, "macro_f1": 0.0})
        return rows
    return score_points


def test_tune_alpha_zero_decoded_once():
    calls = []
    peak = {"switch_ha": 20, "alive_to_dead": 20, "dead_to_alive": 40, "alpha": 2, "floor": 5}
    best, rows = ex.tune_stage(_fake_scorer(peak, calls), ex.STAGE1_GRID, detector=True)
    assert best == {"switch_ha": 20, "alive_to_dead": 20, "dead_to_alive": 40, "alpha": 2, "floor": 5}
    # 3 switch_ha x 3 D pairs x (1 alpha-0 point + 4 alphas x 2 floors), each decoded once;
    # floor 5 is the top of its 2-value axis, so floor 10 is added for the alpha > 0 points.
    assert len(calls) == 3 * 3 * 9 + 3 * 3 * 4
    assert len(calls) == len(set(tuple(sorted(c.items())) for c in calls))
    assert {c["floor"] for c in calls if c["alpha"] == 0} == {1}
    assert len(rows) == len(calls) * 3


def test_tune_no_detector_uses_alpha_zero_only():
    calls = []
    peak = {"switch_ha": 20, "alive_to_dead": 20, "dead_to_alive": 40, "alpha": 8, "floor": 5}
    best, _ = ex.tune_stage(_fake_scorer(peak, calls), ex.STAGE1_GRID, detector=False)
    assert best["alpha"] == 0 and len(calls) == 9


def test_tune_extends_edges_once():
    calls = []
    peak = {"switch_ha": 1000, "alive_to_dead": 5, "dead_to_alive": 10, "alpha": 100, "floor": 0}
    best, _ = ex.tune_stage(_fake_scorer(peak, calls), ex.STAGE1_GRID, detector=True)
    assert best == {"switch_ha": 80, "alive_to_dead": 5, "dead_to_alive": 10, "alpha": 16, "floor": 0.5}
    assert len(calls) == len(set(tuple(sorted(c.items())) for c in calls))
    # extended grid: 4 switch_ha x 4 D pairs x (1 + 5 alphas x 3 floors)
    assert len(calls) == 4 * 4 * 16


def _rec(spell, macro):
    return {"_means": (spell, macro), "mean": {"spell_f1": round(spell, 4), "macro_f1": round(macro, 4)}}


def test_best_of_uses_unrounded_means_and_ties():
    v = {"a": _rec(0.60004, 0.80), "b": _rec(0.60001, 0.90), "c": _rec(0.60004, 0.80)}
    assert ex._best_of(v) == "a"  # rounded means would tie a and b at 0.6000 and pick b on macroF1
    assert ex._ordered(v) == ["a", "c", "b"]


def test_ensemble_member_keys_shift_training_seeds():
    cfg = {"features": True, "detector": True, "k_s": 0.5, "through_dead_s": None,
           **ex.STAGE4_TWEAKS["a_ensemble"]}
    key = ex.Evidence.emit_key(cfg)
    members = ex.Evidence.members(key)
    assert [m[4] for m in members] == [(100, 101, 102), (103, 104, 105), (106, 107, 108)]
    assert [m[5] for m in members] == [0, 1, 2]
    # the incumbent's emitter is member 0 of the ensemble (trained once, shared)
    inc = {k: cfg[k] for k in ("features", "detector", "k_s", "through_dead_s")}
    assert ex.Evidence.members(ex.Evidence.emit_key(inc)) == members[:1]
    assert ex.Evidence.emit_key({**inc, "stride": ex.STRIDE, "hgb": {}}) == ex.Evidence.emit_key(inc)
    d = {**inc, **ex.STAGE4_TWEAKS["d_seeds"]}
    assert ex.Evidence.det_key(d)[3] == tuple(range(100, 106))
    assert ex.Evidence.det_key({**inc, **ex.STAGE4_TWEAKS["b_hgb"]}) == ex.Evidence.det_key(inc)


def test_combine_tweaks_and_runtime_estimates():
    both = ex.combine_tweaks(["a_ensemble", "d_seeds"])
    assert both == {"ensemble_k": 3, "train_seeds": list(range(100, 106))}
    inc = {"emit": 100.0, "det": 50.0}
    assert ex.est_run_seconds(inc, inc) == ex.RUN_SECONDS
    assert ex.est_run_seconds({"emit": 300.0, "det": 50.0}, inc) == pytest.approx(ex.RUN_SECONDS + 2 * ex.FINAL_EMIT_S)
    assert ex.est_run_seconds({"emit": 100.0, "det": 100.0}, inc) == pytest.approx(ex.RUN_SECONDS + ex.FINAL_DET_S)
    assert ex.est_run_seconds({"emit": 100.0, "det": 0.0}, {"emit": 100.0, "det": 0.0}) == ex.RUN_SECONDS
    costs = {"a_ensemble": {"emit": 300.0, "det": 50.0}, "d_seeds": {"emit": 200.0, "det": 100.0}}
    assert ex._prior_cost(["a_ensemble", "d_seeds"], costs, inc) == {"emit": 600.0, "det": 100.0}
