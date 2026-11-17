import numpy as np
import pandas as pd

from bsd.output import write_submission


def test_submission_rows_and_frames(tmp_path):
    n = 199319
    codes = np.random.default_rng(0).integers(0, 4, n).astype(np.int8)
    path = tmp_path / "sub.csv"
    sha = write_submission(codes, path)
    assert isinstance(sha, str) and len(sha) == 64
    with open(path) as fh:
        assert fh.readline().strip() == "frame,label"
    df = pd.read_csv(path, dtype={"label": "object"})
    assert list(df.columns) == ["frame", "label"]
    assert len(df) == n
    assert np.array_equal(df["frame"].to_numpy(), np.arange(n))
    assert set(df["label"]) <= set("HADX")
    assert np.array_equal(df["label"].map("HADX".index).to_numpy(), codes)


def test_submission_sha_is_deterministic(tmp_path):
    codes = np.array([0, 1, 2, 3, 2], dtype=np.int8)
    a = write_submission(codes, tmp_path / "a.csv")
    b = write_submission(codes, tmp_path / "b.csv")
    assert a == b
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
