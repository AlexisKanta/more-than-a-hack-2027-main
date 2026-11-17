import numpy as np
import pytest

import bsd.model as model_mod
from bsd.features import mirror
from bsd.model import train_emitter, visibility_weights

H, A, D, X = 0, 1, 2, 3
NAMES = ["near_home", "near_away", "depth_diff", "speed_all"]


@pytest.fixture(autouse=True)
def small_model(monkeypatch):
    """Fewer boosting rounds and smaller leaves: same code path, a fraction of the time."""
    monkeypatch.setitem(model_mod.HGB_PARAMS, "max_iter", 30)
    monkeypatch.setitem(model_mod.HGB_PARAMS, "min_samples_leaf", 20)


def _game(seed, n=3000):
    """Synthetic frames whose labels follow the features: closer team owns, slow play is dead."""
    rng = np.random.default_rng(seed)
    near_h = rng.gamma(2.0, 2.0, n)
    near_a = rng.gamma(2.0, 2.0, n)
    depth = 5.0 * (near_a - near_h) + rng.normal(0, 3, n)
    speed = rng.uniform(0.2, 3.0, n)
    Xg = np.column_stack([near_h, near_a, depth, speed]).astype(np.float32)
    Xg[rng.random(n) < 0.3, :2] = np.nan  # ball not tracked
    y = np.where(near_h < near_a, H, A).astype(np.int8)
    y[speed < 0.8] = D
    y[rng.random(n) < 0.05] = X
    return Xg, y


def _train(seed=0):
    games = [_game(1), _game(2)]
    return train_emitter([g[0] for g in games], [g[1] for g in games], NAMES, seed=seed)


def test_logp_shape_and_normalisation():
    em = _train()
    Xt, _ = _game(9)
    lp = em.logp(Xt)
    assert lp.shape == (len(Xt), 3)
    assert np.isfinite(lp).all()
    assert np.array_equal(lp, np.round(lp, 6))
    # Undo the prior division: the class probabilities sum to 1.
    p = np.exp(lp) * em.prior
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-4)
    assert em.prior.shape == (3,)
    assert em.prior.sum() == pytest.approx(1.0)
    assert em.prior[H] == pytest.approx(em.prior[A])  # mirrored training is team-balanced
    # The evidence points the right way on clear frames.
    clear_h = (Xt[:, 0] < 1.0) & (Xt[:, 1] > 6.0) & (Xt[:, 3] > 1.5)
    assert clear_h.any() and (lp[clear_h, H] > lp[clear_h, A]).mean() > 0.9


def test_training_deterministic():
    Xt, _ = _game(9)
    np.testing.assert_array_equal(_train().logp(Xt), _train().logp(Xt))


def test_mirror_symmetry():
    em = _train()
    Xt, _ = _game(9)
    lp = em.logp(Xt)
    lpm = em.logp(mirror(Xt, NAMES))
    np.testing.assert_allclose(lpm[:, [A, H, D]], lp, atol=0.05)


def test_training_drops_x_and_leaves_inputs_untouched():
    games = [_game(1), _game(2)]
    copies = [(g[0].copy(), g[1].copy()) for g in games]
    em = train_emitter([g[0] for g in games], [g[1] for g in games], NAMES, seed=0)
    for (Xg, y), (Xc, yc) in zip(games, copies):
        np.testing.assert_array_equal(Xg, Xc)
        np.testing.assert_array_equal(y, yc)
    assert list(em.classes) == [H, A, D]


def test_emit_checks_feature_names():
    em = _train()
    Xt, _ = _game(9)
    np.testing.assert_array_equal(em.emit(Xt, NAMES), em.logp(Xt))
    with pytest.raises(ValueError):
        em.emit(Xt, NAMES[::-1])


def test_visibility_weights_equalise_rate_per_label():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 20_000)
    visible = rng.random(len(y)) < np.array([0.9, 0.8, 0.05])[y]  # D mostly without a ball
    w = visibility_weights(visible, y, 0.6)
    for s in (H, A, D):
        sel = y == s
        assert (w[sel] * visible[sel]).sum() / w[sel].sum() == pytest.approx(0.6)
        assert w[sel].sum() == pytest.approx(sel.sum())  # label totals (and so the prior) kept


def test_visibility_weights_degenerate_label():
    y = np.array([H, H, A, A, D, D])
    visible = np.array([True, False, True, False, False, False])  # D never visible
    w = visibility_weights(visible, y, 0.7)
    np.testing.assert_allclose(w[y == D], 1.0)
    np.testing.assert_allclose(w[:2], [0.7 / 0.5, 0.3 / 0.5])


def _vis_game(seed, n):
    """Labels from speed and proximity; the ball is mostly missing on dead frames (sample-game
    pattern), and ``ball_visible`` says nothing else."""
    rng = np.random.default_rng(seed)
    near_h, near_a = rng.gamma(2.0, 2.0, n), rng.gamma(2.0, 2.0, n)
    speed = rng.uniform(0.2, 3.0, n)
    y = np.where(near_h < near_a, H, A).astype(np.int8)
    y[speed < 1.2] = D
    vis = (rng.random(n) < np.where(y == D, 0.05, 0.9)).astype(np.float32)
    speed_seen = speed + rng.normal(0, 0.8, n)  # noisy, so visibility is worth learning
    return np.column_stack([vis, near_h, near_a, speed_seen]).astype(np.float32), y


VIS_NAMES = ["ball_visible", "near_home", "near_away", "speed_all"]


def _d_shift_from_visibility(em, Xt):
    on, off = Xt.copy(), Xt.copy()
    on[:, 0], off[:, 0] = 1.0, 0.0
    return np.abs(em.logp(on)[:, D] - em.logp(off)[:, D]).mean()


def test_visible_rate_removes_visibility_as_dead_cue():
    """Unweighted, the model learns "no ball => dead"; weighted, most of that cue is gone. (Not
    all of it: the few visible dead rows carry large weights and the trees fit them coarsely.)"""
    games = [_vis_game(1, 20_000), _vis_game(2, 20_000)]
    Xs, ys = [g[0] for g in games], [g[1] for g in games]
    plain = train_emitter(Xs, ys, VIS_NAMES, seed=0)
    weighted = train_emitter(Xs, ys, VIS_NAMES, seed=0, visible_rate=0.6)
    Xt, _ = _vis_game(9, 6000)
    shift_plain = _d_shift_from_visibility(plain, Xt)
    assert shift_plain > 2.0
    assert _d_shift_from_visibility(weighted, Xt) < 0.25 * shift_plain
    with pytest.raises(ValueError):
        train_emitter([g[0] for g in [_game(1)]], [_game(1)[1]], NAMES, visible_rate=0.6)


def test_train_emitter_defaults_unchanged():
    games = [_game(1), _game(2)]
    Xs, ys = [g[0] for g in games], [g[1] for g in games]
    Xt, _ = _game(9)
    plain = train_emitter(Xs, ys, NAMES, seed=0)
    explicit = train_emitter(Xs, ys, NAMES, seed=0, hgb=None, stride=None)
    np.testing.assert_array_equal(plain.logp(Xt), explicit.logp(Xt))
    # Overrides equal to the defaults change nothing; real overrides change the model.
    same = train_emitter(Xs, ys, NAMES, seed=0, hgb={"learning_rate": model_mod.HGB_PARAMS["learning_rate"]},
                         stride=model_mod.STRIDE)
    np.testing.assert_array_equal(plain.logp(Xt), same.logp(Xt))
    for kw in ({"hgb": {"max_iter": 10}}, {"stride": 1}):
        assert not np.array_equal(plain.logp(Xt), train_emitter(Xs, ys, NAMES, seed=0, **kw).logp(Xt))
    assert train_emitter(Xs, ys, NAMES, seed=0, hgb={"max_iter": 10}).model.max_iter == 10
    assert model_mod.HGB_PARAMS["max_iter"] == 30  # the override did not leak into the defaults


def test_ensemble_single_member_equals_emitter():
    em = _train()
    Xt, _ = _game(9)
    ens = model_mod.EnsembleEmitter([em])
    np.testing.assert_array_equal(ens.logp(Xt), em.logp(Xt))
    np.testing.assert_array_equal(ens.emit(Xt, NAMES), em.emit(Xt, NAMES))
    with pytest.raises(ValueError):
        ens.emit(Xt, NAMES[::-1])


def test_ensemble_averages_probabilities_before_log():
    a, b = _train(seed=0), _train(seed=1)
    Xt, _ = _game(9)
    ens = model_mod.EnsembleEmitter((a, b))
    p = np.clip(0.5 * (a.proba(Xt) + b.proba(Xt)), model_mod.MIN_PROBA, 1.0)
    np.testing.assert_allclose(ens.logp(Xt), np.log(p) - np.log(a.prior), atol=1e-6)
    assert np.array_equal(ens.logp(Xt), np.round(ens.logp(Xt), 6))
    with pytest.raises(ValueError):
        model_mod.EnsembleEmitter(())
    other = train_emitter([_game(3)[0]], [_game(3)[1]], NAMES, seed=0)  # different prior
    with pytest.raises(ValueError):
        model_mod.EnsembleEmitter((a, other))


def test_train_emitter_dataset_weights():
    games = [_game(1), _game(2)]
    Xs, ys = [g[0] for g in games], [g[1] for g in games]
    Xt, _ = _game(9)
    plain = train_emitter(Xs, ys, NAMES, seed=0)
    # weights=None and all-ones weights are the default model
    for w in (None, [1.0, 1.0]):
        np.testing.assert_array_equal(plain.logp(Xt), train_emitter(Xs, ys, NAMES, seed=0, weights=w).logp(Xt))
    # a down-weighted dataset changes the model; prior stays the row frequency
    half = train_emitter(Xs, ys, NAMES, seed=0, weights=[1.0, 0.5])
    assert not np.array_equal(plain.logp(Xt), half.logp(Xt))
    np.testing.assert_array_equal(plain.prior, half.prior)
    with pytest.raises(ValueError):
        train_emitter(Xs, ys, NAMES, seed=0, weights=[1.0])


def test_dataset_row_weights_multiply_visibility_weights(monkeypatch):
    captured = {}
    orig = model_mod.HistGradientBoostingClassifier.fit

    def spy(self, X, y, sample_weight=None):
        captured.setdefault("w", []).append(sample_weight)
        return orig(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(model_mod.HistGradientBoostingClassifier, "fit", spy)
    names = NAMES[:3] + ["ball_visible"]
    games = []
    for s in (1, 2):
        Xg, y = _game(s)
        Xg[:, 3] = np.isfinite(Xg[:, 0])
        games.append((Xg, y))
    Xs, ys = [g[0] for g in games], [g[1] for g in games]
    train_emitter(Xs, ys, names, seed=0, visible_rate=0.6)
    train_emitter(Xs, ys, names, seed=0, visible_rate=0.6, weights=[1.0, 0.5])
    w0, w1 = captured["w"]
    n0 = 2 * int((ys[0][::model_mod.STRIDE] != X).sum())  # rows of dataset 0 (+ mirror)
    np.testing.assert_allclose(w1[:n0], w0[:n0])
    np.testing.assert_allclose(w1[n0:], 0.5 * w0[n0:])
