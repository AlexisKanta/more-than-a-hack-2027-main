"""Write a ``frame,label`` submission CSV from int8 label codes (H=0, A=1, D=2, X=3)."""
import hashlib
from pathlib import Path

import numpy as np

LETTERS = np.array(list("HADX"))


def write_submission(codes, path):
    """Write row ``i`` = frame ``i`` with its label letter; return the file's SHA-256 hex digest.

    The bytes are a pure function of ``codes`` (LF line endings, no index column), so the digest
    identifies the submission exactly.
    """
    codes = np.asarray(codes)
    if codes.ndim != 1 or ((codes < 0) | (codes > 3)).any():
        raise ValueError("codes must be a 1-D array of values in 0..3")
    letters = LETTERS[codes.astype(np.intp)]
    body = "".join("%d,%s\n" % (i, s) for i, s in enumerate(letters.tolist()))
    data = ("frame,label\n" + body).encode("ascii")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
