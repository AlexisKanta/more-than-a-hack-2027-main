# Reproducing BSD.csv (team BSD)

One command regenerates the submission byte for byte and checks its hash:

```bash
python -m bsd.run --check-sha aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450
```

The steps below go from a fresh clone to that command. Run everything from the repository root.

## 1. Environment

Tested with **Python 3.14.6** on macOS (arm64). All dependencies are pinned in
`requirements-bsd.txt` (direct and indirect dependencies, exact versions).

```bash
python3.14 -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements-bsd.txt
```

## 2. Data

- **Match bundle (gated, not in this repository).** Unpack it into `data/` as described in
  [DATA.md](DATA.md) and [data/README.md](data/README.md). The pipeline reads only
  `data/tactical/BAR-ATH_tactical_atd.txt`, `data/tactical/BAR-ATH_tactical_atd.xml` and
  `data/tactical/BAR-ATH_tactical_smart_tagging.xml`. Alternatively point `MTAH_DATA_DIR` at the
  folder that holds `tactical/`.
- **Metrica Sports public sample data (training data).** Downloaded from GitHub at the pinned
  commit `e706dd506b360d69d9d123d5b8026e7294b13996` of
  [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) into `data/sample/`
  (files already present are skipped):

  ```bash
  python -m bsd.fetch_sample
  ```

## 3. Run

```bash
python -m bsd.run --check-sha aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450
```

This parses the tracking, trains the per-frame classifier and the hand-over change detector from scratch on the three
sample games (seeded, single-threaded), adds the Smart Tagging evidence, Viterbi-decodes each graded half
with the costs and hand-over settings in the committed `bsd/decode_params.json`, writes `evaluation/results/BSD.csv`,
runs `evaluation/validate_submission.py` on it, and exits non-zero if validation fails or the
file's SHA-256 differs from the value given.

- Runtime: about **4 min 15 s** on an Apple-silicon laptop (single-threaded).
- Expected SHA-256 of `evaluation/results/BSD.csv`:
  `aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450`
- Expected diagnostics: 223 possessions, mean 18.78 s, D share 34.8 % / 33.3 % per half.
- Intermediate results are cached in `cache/` (gitignored). Deleting it forces a full rebuild and
  gives the same file.
- Reference platform: the published SHA-256 was produced and verified (two clean runs gave the same
  hash; the earlier v1 file was also verified from a fresh `git archive` checkout) on macOS arm64 with Python 3.14.6.
  Byte-identity on other platforms is expected but was not verified.

## Optional

```bash
python -m pytest -q                       # unit tests
python -m bsd.bench --mode learned        # leave-one-game-out bench + decoder tuning (~9 min);
                                          # rewrites bsd/decode_params.json
python evaluation/validate_submission.py evaluation/results/BSD.csv
```

`python -m bsd.bench --mode learned` regenerates `bsd/decode_params.json`; the submission depends
on that file, so rerunning the bench is not part of the reproduction.
