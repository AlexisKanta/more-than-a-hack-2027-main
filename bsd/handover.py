"""Hand-over (possession change) labels derived from a per-frame H/A/D/X label array."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from bsd.decode import A, D, H, X
from bsd.features import mirror
from bsd.model import HGB_PARAMS, STRIDE, visibility_weights


def change_labels(labels, periods, fps, k_s=0.5, through_dead_s=None):
    """int8 per frame: 1 near an H<->A change, 0 otherwise, -1 where unscored.

    A change between frames t-1 and t is centred at t-0.5; frame f is 1 iff
    ``|f - (t - 0.5)| <= k`` with ``k = round(k_s * fps)``, i.e. f in ``[t-k, t+k-1]``.
    Changes never span a period boundary. With ``through_dead_s``, an alive-D-alive run whose
    D part lasts at most that many seconds and whose team differs across it also counts, placed
    at the first alive frame after the D run. -1 marks X frames and frames outside ``periods``.
    """
    labels = np.asarray(labels)
    n = len(labels)
    k = int(round(k_s * fps))
    out = np.full(n, -1, dtype=np.int8)
    max_dead = None if through_dead_s is None else through_dead_s * fps
    for _, a, b in periods:
        a, b = int(a), min(int(b) + 1, n)
        seg = labels[a:b]
        out[a:b] = np.where(seg == X, -1, 0)
        times = []
        alive = np.flatnonzero((seg == H) | (seg == A))
        if len(alive) > 1:
            prev, cur = alive[:-1], alive[1:]
            gap = cur - prev
            differ = seg[prev] != seg[cur]
            direct = differ & (gap == 1)
            times.extend((cur[direct] + a).tolist())
            if max_dead is not None:
                between = differ & (gap > 1) & (gap - 1 <= max_dead)
                for p, c in zip(prev[between], cur[between]):
                    if (seg[p + 1:c] == D).all():
                        times.append(int(c) + a)
        for t in times:
            lo, hi = max(t - k, a), min(t + k, b)
            win = out[lo:hi]
            win[win == 0] = 1
    return out


@dataclass(frozen=True)
class Detector:
    """Binary "possession changes near this frame" classifier (team-symmetric)."""
    model: HistGradientBoostingClassifier
    names: tuple

    def p_change(self, X, names):
        """(F,) probability of class 1: mean over X and its mirror, rounded to 6 decimals."""
        if list(names) != list(self.names):
            raise ValueError("feature names differ from the ones the detector was trained on")
        X = np.asarray(X, dtype=np.float32)
        with threadpool_limits(limits=1):
            p = self.model.predict_proba(X)[:, 1]
            pm = self.model.predict_proba(mirror(X, list(self.names)))[:, 1]
        return np.round(0.5 * (p + pm), 6)


def train_detector(Xs, cs, names, seed=0, visible_rate=None):
    """Fit on every ``STRIDE``-th frame of each (X, change-label) pair plus its mirror.

    Rows labelled -1 are dropped; a change is team-symmetric so mirrored copies keep their label.
    """
    names = list(names)
    if visible_rate is not None and "ball_visible" not in names:
        raise ValueError("visible_rate needs a ball_visible feature column")
    feats, labels = [], []
    for Xg, c in zip(Xs, cs):
        Xg = np.asarray(Xg, dtype=np.float32)[::STRIDE]
        c = np.asarray(c)[::STRIDE]
        keep = c >= 0
        Xk, ck = Xg[keep], c[keep].astype(np.int64)
        feats += [Xk, mirror(Xk, names)]
        labels += [ck, ck]
    Xt = np.concatenate(feats)
    yt = np.concatenate(labels)
    if len(np.unique(yt)) < 2:
        raise ValueError("training labels must contain both change and no-change frames")
    weights = None
    if visible_rate is not None:
        weights = visibility_weights(Xt[:, names.index("ball_visible")] > 0, yt, visible_rate)
    model = HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS)
    with threadpool_limits(limits=1):
        model.fit(Xt, yt, sample_weight=weights)
    return Detector(model, tuple(names))
