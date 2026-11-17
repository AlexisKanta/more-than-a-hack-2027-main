"""Learned per-frame H/A/D evidence: a gradient-boosted classifier on ``frame_features``.

``train_emitter`` fits ``HistGradientBoostingClassifier`` on labelled frames (sample games after
``bsd.degrade``). Every frame is used twice, as is and mirrored (home/away columns swapped, H/A
labels swapped), so the model has no built-in home preference; ``Emitter.proba`` also averages
the prediction of a frame with the swapped prediction of its mirror, which makes the evidence
exactly team-symmetric. ``Emitter.logp`` is ``log(p / prior)`` with the training class
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


@dataclass(frozen=True)
class Emitter:
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

    def logp(self, X):
        """(F, 3) log(p / prior), rounded to 6 decimals."""
        p = np.clip(self.proba(X), MIN_PROBA, 1.0)
        return np.round(np.log(p) - np.log(self.prior), 6)

    def emit(self, X, names):
        """``bench`` emitter signature; refuses features in a different column order."""
        if list(names) != list(self.names):
            raise ValueError("feature names differ from the ones the emitter was trained on")
        return self.logp(X)


def train_emitter(Xs, ys, names, seed=0):
    """Fit on every ``STRIDE``-th frame of each (X, y) pair plus its mirror; X-label frames dropped."""
    names = list(names)
    feats, labels = [], []
    for Xg, y in zip(Xs, ys):
        Xg = np.asarray(Xg, dtype=np.float32)[::STRIDE]
        y = np.asarray(y)[::STRIDE]
        keep = y != X
        Xk, yk = Xg[keep], y[keep].astype(np.int64)
        feats += [Xk, mirror(Xk, names)]
        labels += [yk, SWAP[yk]]
    Xt = np.concatenate(feats)
    yt = np.concatenate(labels)
    counts = np.bincount(yt, minlength=3)
    if (counts == 0).any():
        raise ValueError("training labels must contain H, A and D frames")
    model = HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS)
    with threadpool_limits(limits=1):
        model.fit(Xt, yt)
    return Emitter(model, tuple(names), counts / counts.sum())
