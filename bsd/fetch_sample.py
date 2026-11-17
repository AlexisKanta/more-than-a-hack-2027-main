"""Download the public Metrica sample games (pinned commit) into data/sample/."""
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root, same as bsd/events.py

SAMPLE_SHA = "e706dd506b360d69d9d123d5b8026e7294b13996"
_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/%s/data/" % SAMPLE_SHA

_FILES = []
for _g in (1, 2):
    _d = "Sample_Game_%d" % _g
    _FILES += [
        "%s/%s_RawEventsData.csv" % (_d, _d),
        "%s/%s_RawTrackingData_Home_Team.csv" % (_d, _d),
        "%s/%s_RawTrackingData_Away_Team.csv" % (_d, _d),
    ]
_FILES += [
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
]


def sample_urls():
    return {f: _BASE + f for f in _FILES}


def fetch(dest=None):
    dest = str(ROOT / "data" / "sample") if dest is None else str(dest)
    for rel, url in sample_urls().items():
        path = os.path.join(dest, rel)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)


if __name__ == "__main__":
    fetch()
