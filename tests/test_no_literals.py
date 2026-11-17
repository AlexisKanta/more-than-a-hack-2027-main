import re
from pathlib import Path

BSD = Path(__file__).resolve().parent.parent / "bsd"


def test_no_hardcoded_frames():
    pat = re.compile(r"\b\d{5,6}\b")
    hits = []
    for py in sorted(BSD.glob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append("%s:%d: %s" % (py.name, i, line.strip()))
    assert not hits, "5-6 digit literals in bsd/:\n" + "\n".join(hits)
