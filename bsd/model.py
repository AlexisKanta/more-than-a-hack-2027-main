"""Learned per-frame H/A/D evidence: a gradient-boosted classifier on ``frame_features``.

``train_emitter`` fits ``HistGradientBoostingClassifier`` on labelled frames (sample games after
``bsd.degrade``). Every frame is used twice, as is and mirrored (home/away columns swapped, H/A
labels swapped), so the model has no built-in home preference; ``Emitter.proba`` also averages
the prediction of a frame with the swapped prediction of its mirror, which makes the evidence
exactly team-symmetric. With ``visible_rate`` given, rows are weighted so that the ball is
visible on that fraction of each label's rows (``visibility_weights``): in the degraded sample
games the ball is missing on almost every dead frame, which on BAR-ATH (missing ball = mostly
occlusion) would be a false "dead" cue. ``Emitter.logp`` is ``log(p / prior)`` with the training class
frequencies as prior, i.e. a scaled likelihood -- the per-frame evidence the Viterbi decoder
adds up -- rounded to 6 decimals.

Determinism: fixed ``max_iter`` without early stopping, a fixed ``random_state`` and a single
thread (``threadpoolctl``), so a training run gives the same model on any machine.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from bsd.decode import A, D, H, X
from bsd.features import mirror

STRIDE = 3  # train on every STRIDE-th frame: neighbouring frames are near-duplicates
MIN_PROBA = 1e-6
HGB_PARAMS = dict(
    max_iter=150,
    learning_rate=0.1,
    max_leaf_nodes=31,
    min_samples_leaf=100,
    l2_regularization=1.0,
    early_stopping=False,
)
SWAP = np.array([A, H, D])  # column / label permutation for mirroring


class _LogScores:
    """``logp`` / ``emit`` on top of ``proba``, ``prior`` and ``names`` (shared by both emitters)."""

    def logp(self, X):
        """(F, 3) log(p / prior), rounded to 6 decimals."""
        p = np.clip(self.proba(X), MIN_PROBA, 1.0)
        return np.round(np.log(p) - np.log(self.prior), 6)

    def emit(self, X, names):
        """``bench`` emitter signature; refuses features in a different column order."""
        if list(names) != list(self.names):
            raise ValueError("feature names differ from the ones the emitter was trained on")
        return self.logp(X)


@dataclass(frozen=True)
class Emitter(_LogScores):
    model: HistGradientBoostingClassifier
    names: tuple
    prior: np.ndarray  # (3,) training frequencies of H, A, D

    @property
    def classes(self):
        return self.model.classes_

    def proba(self, X):
        """(F, 3) team-symmetric class probabilities, columns H, A, D."""
        X = np.asarray(X, dtype=np.float32)
        with threadpool_limits(limits=1):
            p = self.model.predict_proba(X)
            pm = self.model.predict_proba(mirror(X, list(self.names)))
        return 0.5 * (p + pm[:, SWAP])


@dataclass(frozen=True)
class EnsembleEmitter(_LogScores):
    """Mean of the members' (mirror-averaged) probabilities, then ``log(p / prior)`` as in
    ``Emitter.logp`` with the members' common prior. Members must share feature names and prior."""
    emitters: tuple

    def __post_init__(self):
        ems = tuple(self.emitters)
        if not ems:
            raise ValueError("an ensemble needs at least one emitter")
        if any(list(e.names) != list(ems[0].names) for e in ems):
            raise ValueError("ensemble members were trained on different feature names")
        if any(not np.array_equal(e.prior, ems[0].prior) for e in ems):
            raise ValueError("ensemble members have different priors")
        object.__setattr__(self, "emitters", ems)

    @property
    def names(self):
        return self.emitters[0].names

    @property
    def prior(self):
        return self.emitters[0].prior

    def proba(self, X):
        """(F, 3) mean of the members' team-symmetric probabilities."""
        return sum(e.proba(X) for e in self.emitters) / len(self.emitters)


def visibility_weights(visible, y, visible_rate):
    """Per-row weights making P(ball visible | label) = ``visible_rate`` for every label.

    For label s with visible fraction q_s: visible rows get ``visible_rate / q_s``, missing rows
    ``(1 - visible_rate) / (1 - q_s)``; each label's weights still sum to its row count, so the
    class prior is unchanged. A label whose rows are all visible or all missing keeps weight 1.
    """
    visible = np.asarray(visible, dtype=bool)
    y = np.asarray(y)
    w = np.ones(len(y), dtype=np.float64)
    for s in np.unique(y):
        sel = y == s
        q = visible[sel].mean()
        if 0.0 < q < 1.0:
            w[sel] = np.where(visible[sel], visible_rate / q, (1.0 - visible_rate) / (1.0 - q))
    return w


def train_emitter(Xs, ys, names, seed=0, visible_rate=None, hgb=None, stride=None):
    """Fit on every ``STRIDE``-th frame of each (X, y) pair plus its mirror; X-label frames dropped.

    ``visible_rate`` (e.g. BAR-ATH's in-period ball-visible fraction) turns on
    ``visibility_weights`` computed from the ``ball_visible`` column. ``hgb`` (a dict merged over
    ``HGB_PARAMS``) and ``stride`` (instead of ``STRIDE``) are overrides for experiments; the
    defaults (None) train exactly the default model.
    """
    names = list(names)
    stride = STRIDE if stride is None else int(stride)
    if visible_rate is not None and "ball_visible" not in names:
        raise ValueError("visible_rate needs a ball_visible feature column")
    feats, labels = [], []
    for Xg, y in zip(Xs, ys):
        Xg = np.asarray(Xg, dtype=np.float32)[::stride]
        y = np.asarray(y)[::stride]
        keep = y != X
        Xk, yk = Xg[keep], y[keep].astype(np.int64)
        feats += [Xk, mirror(Xk, names)]
        labels += [yk, SWAP[yk]]
    Xt = np.concatenate(feats)
    yt = np.concatenate(labels)
    counts = np.bincount(yt, minlength=3)
    if (counts == 0).any():
        raise ValueError("training labels must contain H, A and D frames")
    weights = None
    if visible_rate is not None:
        weights = visibility_weights(Xt[:, names.index("ball_visible")] > 0, yt, visible_rate)
    model = HistGradientBoostingClassifier(random_state=seed, **{**HGB_PARAMS, **(hgb or {})})
    with threadpool_limits(limits=1):
        model.fit(Xt, yt, sample_weight=weights)
    return Emitter(model, tuple(names), counts / counts.sum())
