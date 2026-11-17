"""Download the public DFL/IDSSE Bundesliga open data (Bassek et al., Sci Data 2025, CC BY 4.0).

figshare article 28196177: 7 matches x (matchinformation, events_raw, positions_raw_observed).
Files land in data/dfl/ and are MD5-verified against the figshare API listing.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DFL_ARTICLE = 28196177
_URL = "https://ndownloader.figshare.com/files/%d"

_KINDS = {
    "matchinformation": "DFL_02_01_matchinformation",
    "events": "DFL_03_02_events_raw",
    "positions": "DFL_04_03_positions_raw_observed",
}

# (file_id, kind, competition number, match id, md5) from the figshare API listing.
_TABLE = (
    (51643472, "matchinformation", 1, "J03WN1", "37b0f85588c0acfb995fe633c7ac2e0d"),
    (51643475, "matchinformation", 1, "J03WMX", "bc5d5b21dbcf36aa30f4cd0428309f6b"),
    (51643478, "matchinformation", 2, "J03WOH", "b9710781d400dd812f62b0a6e3d7bce8"),
    (51643481, "matchinformation", 2, "J03WOY", "7b021e577fca917a63111d55cc29ae5b"),
    (51643484, "matchinformation", 2, "J03WQQ", "5979209af316a914ce57347d0633df6c"),
    (51643487, "matchinformation", 2, "J03WPY", "65a4568e4f374af94d3c1015cbc7c4f5"),
    (51643490, "matchinformation", 2, "J03WR9", "5e2d580d3d8ed97e32959de90d04992b"),
    (51643493, "events", 1, "J03WMX", "2d88e180a1f63c7e4157feb980fc7255"),
    (51643496, "events", 1, "J03WN1", "83e8ed7ea8e8ebbbdc61f61ac3534008"),
    (51643499, "events", 2, "J03WOH", "ab66e51512f39d06f49f27dd3b4c2f40"),
    (51643502, "events", 2, "J03WOY", "64b1be01d6ff6848168efebd56f344b9"),
    (51643505, "events", 2, "J03WPY", "f68482441d38723d69f244f896482803"),
    (51643508, "events", 2, "J03WQQ", "b7e9d82af4034539d802319be6c57776"),
    (51643511, "events", 2, "J03WR9", "c13bc5efa5eca4a022cc678210b87d5f"),
    (51643514, "positions", 1, "J03WMX", "e4273a2adf5dbd9d4ec42def2f01905c"),
    (51643517, "positions", 1, "J03WN1", "bacb2f609c1d1b0bfc906f6b9d4109f2"),
    (51643520, "positions", 2, "J03WOH", "3efff25a86a61768e7d253b4bba137a3"),
    (51643523, "positions", 2, "J03WOY", "4e1063e0a0ae499920cf68ac1aa3ab4e"),
    (51643526, "positions", 2, "J03WPY", "e56ff1336a334f8da51110fbe7cab764"),
    (51643529, "positions", 2, "J03WQQ", "9105bad987191d3c3c362c1fce9a4fcd"),
    (51643532, "positions", 2, "J03WR9", "e060bb7d7a5a3bfbd15bbf9bc631024d"),
)


def _name(kind, comp, match):
    return "%s_DFL-COM-%s_DFL-MAT-%s.xml" % (_KINDS[kind], str(comp).zfill(6), match)


# (file_id, name, md5) for all 21 files.
DFL_FILES = tuple((i, _name(k, c, m), md5) for i, k, c, m, md5 in _TABLE)
MATCH_IDS = tuple(sorted({m for _, _, _, m, _ in _TABLE}))


def _short(match_id):
    return match_id.replace("DFL-MAT-", "")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, md5):
    """Raise ValueError unless the file's MD5 equals ``md5``."""
    got = md5_of(path)
    if got != md5:
        raise ValueError("MD5 mismatch for %s: expected %s, got %s" % (path, md5, got))


def _download(url, path):
    urllib.request.urlretrieve(url, path)


def dfl_dir():
    return ROOT / "data" / "dfl"


def fetch(dest=None, matches=None):
    """Download missing files to ``dest`` (default data/dfl/); MD5-check each new download.

    ``matches`` optionally restricts to match ids such as ``["J03WMX"]``. Existing files are skipped.
    """
    dest = str(dfl_dir()) if dest is None else str(dest)
    want = None if matches is None else {_short(m) for m in matches}
    os.makedirs(dest, exist_ok=True)
    for file_id, name, md5 in DFL_FILES:
        if want is not None and not any(name.endswith("_DFL-MAT-%s.xml" % m) for m in want):
            continue
        path = os.path.join(dest, name)
        if os.path.exists(path):
            continue
        tmp = path + ".part"
        _download(_URL % file_id, tmp)
        try:
            verify(tmp, md5)
        except ValueError:
            os.remove(tmp)
            raise
        os.replace(tmp, path)


if __name__ == "__main__":
    fetch()
