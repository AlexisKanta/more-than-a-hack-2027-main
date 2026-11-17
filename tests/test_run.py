import pytest

from bsd.run import check_sha

SHA = "c22cc48bd24841d9dffb61f04aa1df26fac3647d17bb8a676c9f9d5d840f9eee"


def test_check_sha_match_passes():
    check_sha(SHA, SHA)
    check_sha(SHA, "  " + SHA.upper() + "\n")  # case and surrounding whitespace are ignored


def test_check_sha_mismatch_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        check_sha(SHA, "0" * 64)
    assert exc.value.code not in (0, None)
    assert "0" * 64 in str(exc.value.code) and SHA in str(exc.value.code)


# --- wiring of the hand-over block in main (synthetic stand-ins, no BAR-ATH data) ---------------

N = 10
HO = {"features": True, "detector": True, "k_s": 0.5, "through_dead_s": None, "alpha": 16.0, "floor": 0.5}


def _wire(monkeypatch, tmp_path, ho):
    import types

    import numpy as np

    import bsd.bench as bench_mod
    import bsd.decode as decode_mod
    import bsd.features as features_mod
    import bsd.io_epts as io_mod
    import bsd.output as output_mod
    import bsd.run as run_mod
    from bsd.decode import DecodeParams
    from bsd.tracking import Tracking

    seen = {"feature_calls": [], "train_calls": []}
    tr = Tracking(np.zeros((N, 2, 2), np.float32), np.zeros(2, np.int8), np.zeros((N, 2), np.float32),
                  [(1, 0, N - 1)], (105.0, 68.0))
    names = ["a", "b"]

    def fake_features(t, handover=False):
        seen["feature_calls"].append(handover)
        return np.zeros((N, 2), np.float32), names

    class FakeEmitter:
        def emit(self, X, nm):
            return np.zeros((N, 3))

    class FakeDetector:
        def p_change(self, X, nm):
            return np.full(N, 0.25)

    def fake_training(handover=False, meta=False):
        seen["train_calls"].append((handover, meta))
        return [(np.zeros((N, 2), np.float32), np.zeros(N), names, [(1, 0, N - 1)], N, 25)]

    def fake_decode(logp, p, periods, n, switch_cost=None):
        seen["switch_cost"] = switch_cost
        return np.zeros(n, dtype=np.int8)

    params = DecodeParams(10.0, 40.0, 80.0)
    monkeypatch.setattr(bench_mod, "load_decode_params", lambda: ("learned", {"learned": params, "heuristic": params}))
    monkeypatch.setattr(bench_mod, "load_handover_config", lambda mode="learned": ho)
    monkeypatch.setattr(bench_mod, "training_data", fake_training)
    monkeypatch.setattr(bench_mod, "learned_emitter", lambda train: FakeEmitter())
    monkeypatch.setattr(bench_mod, "learned_detector", lambda train, nm, k, t: FakeDetector())
    monkeypatch.setattr(io_mod, "bar_ath_tracking", lambda: tr)
    monkeypatch.setattr(features_mod, "frame_features", fake_features)
    monkeypatch.setattr(decode_mod, "decode_periods", fake_decode)
    monkeypatch.setattr(output_mod, "write_submission", lambda codes, path: "0" * 64)
    monkeypatch.setattr(run_mod, "ROOT", tmp_path)
    monkeypatch.setattr(run_mod, "OUT_PATH", tmp_path / "BSD.csv")
    monkeypatch.setattr(run_mod, "summarize", lambda *a, **k: None)
    monkeypatch.setattr(run_mod.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
    import games

    monkeypatch.setattr(games, "half_video_blocks", lambda g: [(1, 0, N - 1)])
    return run_mod, seen


def test_main_with_handover_block_passes_switch_cost(monkeypatch, tmp_path):
    import numpy as np

    run_mod, seen = _wire(monkeypatch, tmp_path, HO)
    assert run_mod.main(["--no-tagging"]) == 0
    assert seen["feature_calls"] == [True]
    assert seen["train_calls"] == [(True, True)]
    sc = np.asarray(seen["switch_cost"])
    assert sc.shape == (N,) and np.isfinite(sc).all()
    assert np.allclose(sc, 10.0 + 16.0 * np.log(3.0), atol=1e-3)  # p = 0.25, switch_ha 10, alpha 16


def test_main_without_handover_block_is_v1(monkeypatch, tmp_path):
    run_mod, seen = _wire(monkeypatch, tmp_path, None)
    assert run_mod.main(["--no-tagging"]) == 0
    assert seen["feature_calls"] == [False]
    assert seen["train_calls"] == [(False, False)]
    assert seen["switch_cost"] is None
