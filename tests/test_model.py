import numpy as np
import pytest

import bsd.model as model_mod
from bsd.features import mirror
from bsd.model import train_emitter

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
