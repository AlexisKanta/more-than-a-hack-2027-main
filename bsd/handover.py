"""Hand-over (possession change) labels derived from a per-frame H/A/D/X label array."""
import numpy as np

from bsd.decode import A, D, H, X


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
