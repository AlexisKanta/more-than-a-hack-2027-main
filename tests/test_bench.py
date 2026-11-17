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
