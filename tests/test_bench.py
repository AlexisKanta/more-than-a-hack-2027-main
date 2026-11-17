import numpy as np
import pytest

import bsd.bench as bench_mod
from bsd.bench import bench, score
from bsd.decode import DecodeParams
from bsd.tracking import Tracking

H, A, D, X = 0, 1, 2, 3


def _gt():
    # X head (unscored), then H 100, D 50, A 100, D 50, H 100, then X tail.
    return np.array([X] * 20 + [H] * 100 + [D] * 50 + [A] * 100 + [D] * 50 + [H] * 100 + [X] * 30,
                    dtype=np.int8)


def test_score_perfect():
    y = _gt()
    s = score(y, y)
    assert s["spell_f1"] == 1.0
    assert s["macro_f1"] == 1.0
    assert s["n_poss"] == s["gt_n_poss"] == 3
    assert s["mean_poss_s"] == pytest.approx(100 / 25)
    assert s["gt_mean_poss_s"] == pytest.approx(100 / 25)


def test_score_constant_zero():
    y = _gt()
    s = score(y, np.zeros_like(y))
    # One 400-frame H spell: best IoU 100/400 = 0.25 < 0.3 -> nothing matched.
    assert s["spell_f1"] == 0.0
    assert s["n_poss"] == 1 and s["gt_n_poss"] == 3
    assert s["mean_poss_s"] == pytest.approx(400 / 25)
    # Per-frame: H F1 = 2*0.5*1/1.5, A and D F1 = 0.
    assert s["macro_f1"] == pytest.approx((2 / 3) / 3)


def _fake_game(n, length):
    """Tiny game whose labels identify it: label code n % 2 everywhere inside one period."""
    tr = Tracking(np.zeros((length, 2, 2), np.float32), np.array([0, 1], np.int8),
                  np.zeros((length, 2), np.float32), [(1, 5, length - 6)], (105.0, 68.0))
    y = np.full(length, X, np.int8)
    y[5:length - 5] = n % 2
    return tr, y


@pytest.fixture
def fake_data(monkeypatch):
    lengths = {1: 200, 2: 300, 3: 400}
    calls = []
    monkeypatch.setattr(bench_mod, "sample_game", lambda n: _fake_game(n, lengths[n]))
    monkeypatch.setattr(bench_mod, "gap_profile", lambda: "profile")

    def fake_degrade(tr, profile, seed):
        assert profile == "profile"
        calls.append(seed)
        return tr

    monkeypatch.setattr(bench_mod, "degrade", fake_degrade)
    monkeypatch.setattr(bench_mod, "frame_features",
                        lambda tr: (np.zeros((len(tr.ball), 1), np.float32), ["f"]))
    return lengths, calls


def test_bench_leave_one_game_out(fake_data):
    lengths, calls = fake_data
    seen = []

    def make_emit(train):
        seen.append(sorted(len(Xt) for Xt, _, _ in train))
        for Xt, yt, names in train:
            assert len(Xt) == len(yt) and names == ["f"]

        def emit(Xg, names):
            lp = np.zeros((len(Xg), 3))
            lp[:, H] = 1.0  # always home
            return lp
        return emit

    p = DecodeParams(switch_ha=10.0, alive_to_dead=10.0, dead_to_alive=10.0)
    df = bench(make_emit, p, seeds=(0, 1))
    assert list(df["game"]) == [1, 2, 3, 1, 2, 3]
    assert list(df["seed"]) == [0, 0, 0, 1, 1, 1]
    for col in ("spell_f1", "macro_f1", "n_poss", "mean_poss_s", "gt_n_poss", "gt_mean_poss_s"):
        assert col in df.columns
    # The training set never contains the held-out game.
    held = {1: [300, 400], 2: [200, 400], 3: [200, 300]}
    assert seen == [held[g] for g in (1, 2, 3)] * 2
    assert sorted(set(calls)) == [0, 1]
    # Game 2 is all H inside its period -> an always-H emitter is perfect there; games 1, 3 are A.
    assert (df.loc[df.game == 2, "spell_f1"] == 1.0).all()
    assert (df.loc[df.game != 2, "spell_f1"] == 0.0).all()


def test_heldout_logp_trains_once_per_fold_on_train_seeds(fake_data):
    lengths, calls = fake_data
    seen = []

    def make_emit(train):
        seen.append(sorted(len(Xt) for Xt, _, _ in train))
        return lambda Xg, names: np.zeros((len(Xg), 3))

    held, pool = bench_mod.prepare(seeds=(0, 1), train_seeds=(100, 101, 102))
    cases = bench_mod.heldout_logp(make_emit, held, pool)
    assert [(c["game"], c["seed"]) for c in cases] == [(g, s) for s in (0, 1) for g in (1, 2, 3)]
    assert all(c["logp"].shape == (lengths[c["game"]], 3) for c in cases)
    # One training per held-out game, on the other two games x 3 training seeds.
    held_out = {1: [300] * 3 + [400] * 3, 2: [200] * 3 + [400] * 3, 3: [200] * 3 + [300] * 3}
    assert seen == [held_out[g] for g in (1, 2, 3)]
    assert sorted(set(calls)) == [0, 1, 100, 101, 102]


def _fake_tuning(monkeypatch, peak):
    """decode returns the params; score is a function of them peaking at ``peak``."""
    monkeypatch.setattr(bench_mod, "decode_periods", lambda logp, p, periods, n: p)

    def fake_score(labels, p):
        v = -sum(abs(np.log2(getattr(p, k) / peak[k])) for k in peak)
        return {"spell_f1": v, "macro_f1": 0.0, "n_poss": 1, "mean_poss_s": 1.0,
                "gt_n_poss": 1, "gt_mean_poss_s": 1.0}

    monkeypatch.setattr(bench_mod, "score", fake_score)
    return [{"game": g, "seed": 0, "logp": None, "labels": None, "periods": None, "n": 0}
            for g in (1, 2, 3)]


def test_tune_decode_interior_optimum(monkeypatch):
    cases = _fake_tuning(monkeypatch, {"switch_ha": 20, "alive_to_dead": 10, "dead_to_alive": 20})
    p, grid = bench_mod.tune_decode(cases)
    assert p == DecodeParams(switch_ha=20, alive_to_dead=10, dead_to_alive=20)
    assert len(grid) == 5 * 4 * 4 * 3  # every grid point x every case, no extension


def test_tune_decode_extends_edges_once(monkeypatch):
    # Optimum beyond the top of switch_ha and below the bottom of alive_to_dead.
    cases = _fake_tuning(monkeypatch, {"switch_ha": 1000, "alive_to_dead": 1, "dead_to_alive": 20})
    p, grid = bench_mod.tune_decode(cases)
    assert p == DecodeParams(switch_ha=160, alive_to_dead=2.5, dead_to_alive=20)
    assert sorted(grid["switch_ha"].unique()) == [5, 10, 20, 40, 80, 160]
    assert sorted(grid["alive_to_dead"].unique()) == [2.5, 5, 10, 20, 40]
    assert len(grid) == 6 * 5 * 4 * 3  # each point decoded once


def test_tune_decode_tie_break_prefers_macro_then_grid_order(monkeypatch):
    monkeypatch.setattr(bench_mod, "decode_periods", lambda logp, p, periods, n: p)
    monkeypatch.setattr(bench_mod, "score", lambda labels, p: {
        "spell_f1": 0.5, "macro_f1": 1.0 if p.dead_to_alive == 40 else 0.0, "n_poss": 1,
        "mean_poss_s": 1.0, "gt_n_poss": 1, "gt_mean_poss_s": 1.0})
    cases = [{"game": 1, "seed": 0, "logp": None, "labels": None, "periods": None, "n": 0}]
    grid = {"switch_ha": (10, 20, 30), "alive_to_dead": (10, 20, 30), "dead_to_alive": (20, 40, 80)}
    p, _ = bench_mod.tune_decode(cases, grid)
    assert p == DecodeParams(switch_ha=10, alive_to_dead=10, dead_to_alive=40)


def test_load_decode_params(tmp_path):
    rec = {"adopted": "learned",
           "heuristic": {"switch_ha": 40.0, "alive_to_dead": 10.0, "dead_to_alive": 20.0, "bench_spell_f1": 0.3},
           "learned": {"switch_ha": 80.0, "alive_to_dead": 5.0, "dead_to_alive": 2.5, "bench_spell_f1": 0.5}}
    path = tmp_path / "p.json"
    path.write_text(__import__("json").dumps(rec))
    adopted, tuned = bench_mod.load_decode_params(path)
    assert adopted == "learned"
    assert tuned["heuristic"] == DecodeParams(40.0, 10.0, 20.0)
    assert tuned["learned"] == DecodeParams(80.0, 5.0, 2.5)


def test_load_handover_config(tmp_path):
    import json

    ho = {"features": True, "detector": True, "k_s": 0.5, "through_dead_s": None, "alpha": 16.0, "floor": 0.5}
    rec = {"adopted": "learned",
           "heuristic": {"switch_ha": 40.0, "alive_to_dead": 10.0, "dead_to_alive": 20.0},
           "learned": {"switch_ha": 10.0, "alive_to_dead": 40.0, "dead_to_alive": 80.0, "handover": ho}}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(rec))
    assert bench_mod.load_handover_config(path) == ho
    assert bench_mod.load_handover_config(path, "heuristic") is None
    adopted, tuned = bench_mod.load_decode_params(path)  # the extra block does not disturb the costs
    assert tuned["learned"] == DecodeParams(10.0, 40.0, 80.0)
    del rec["learned"]["handover"]
    path.write_text(json.dumps(rec))
    assert bench_mod.load_handover_config(path) is None


def test_committed_params_have_stage1_handover():
    ho = bench_mod.load_handover_config()
    assert ho["features"] and ho["detector"] and ho["alpha"] == 16.0 and ho["floor"] == 0.5


def test_training_data_meta_shapes(monkeypatch):
    seen = []

    def fake(g, profile, s, handover=False, with_fps=False):
        seen.append((g, s, handover, with_fps))
        return ("X", "y", "names", "periods", 5, 25) if with_fps else ("X", "y", "names", "periods", 5)

    monkeypatch.setattr(bench_mod, "_degraded", fake)
    monkeypatch.setattr(bench_mod, "gap_profile", lambda: {})
    plain = bench_mod.training_data(games=(1,), seeds=(7,))
    full = bench_mod.training_data(games=(1,), seeds=(7,), handover=True, meta=True)
    assert plain == [("X", "y", "names")] and full == [("X", "y", "names", "periods", 5, 25)]
    assert seen == [(1, 7, False, False), (1, 7, True, True)]
