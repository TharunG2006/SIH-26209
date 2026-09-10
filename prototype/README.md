# Explainable Satellite Anomaly Detection — Prototype

SIH 2026, PS 26209. Working prototype for the AI-based satellite health
monitoring system described in `../CLAUDE.md`.

## What it does

Learns each spacecraft's normal telemetry behaviour with an LSTM, flags
timesteps where reality diverges from the forecast, and — the part that
distinguishes it from the NASA baseline — **explains every alert by naming the
channels and subsystems that caused it**, ranked by how much each contributed
and ordered by which moved first.

## Headline numbers

Selected cross-mission: every setting is chosen on one spacecraft and reported
on the other, so nothing below is tuned on what it is scored against.

| | SMAP | MSL | telemanom (NASA) |
|---|---|---|---|
| precision | **0.868** | 0.514 | 0.855 / 0.926 |
| recall | 0.603 | 0.472 | 0.855 / 0.694 |
| top-1 attribution | **1.00** (n=19) | 0.857 (n=7) | *cannot report* |

Pooled attribution **25/26 = 0.962**, 95% CI [0.81, 0.99]. NASA's baseline
produces a score, not an attribution, so the last row has nothing to compare
against — that is the gap this project exists to fill.

Against a conventional fixed-limit alarm we are **never earlier**, and we fire
on **0.04%** of healthy readings where that limit fires on 8.6% (SMAP) and 19.1%
(MSL). The claim is comparable coverage two orders of magnitude quieter, with an
explanation attached — not earlier detection, and not prediction. See
`reports/RESULTS.md`.

## Running it

There is no separate frontend and backend to start. Streamlit is both the web
server and the UI, it imports `src/` directly rather than calling an API, and
the database is a SQLite file with no server process. One command runs the whole
application:

```bash
python -m streamlit run prototype/dashboard/app.py
```

That serves the dashboard on http://localhost:8501.

### First time on a new machine

```bash
pip install -r prototype/requirements.txt
python prototype/src/fetch_data.py          # NASA SMAP/MSL telemetry (~9 MB)
python prototype/src/train.py               # joint model, ~12 min
python prototype/src/train_channels.py      # 81 per-channel models, ~70 min
python prototype/src/populate_db.py         # fill the SQLite schema
python -m streamlit run prototype/dashboard/app.py
```

Training is the slow part and only needs doing once; everything afterwards reads
the saved checkpoints. `train_channels.py` is resumable — rerun it after an
interruption and it skips the channels it already finished. Scripts run from any
directory.

### The rest of the command line

```bash
python prototype/src/evaluate.py --sweep       # score against NASA's labels
python prototype/src/tune.py                   # cross-mission tuning, held-out results
python prototype/src/baseline_limits.py        # us vs a fixed-limit alarm
python prototype/src/baseline_cost.py          # what that limit costs in false alarms
python prototype/src/early.py                  # early-warning lift (negative result)
python prototype/src/precursor.py              # model-free precursor test
python prototype/src/run_esa_precursor.py      # the same test on ESA's 1,203 anomalies
python prototype/src/sensors.py                # what each live channel measures
python prototype/src/satnogs.py --list         # live satellites with a decoder
python prototype/src/satnogs.py --norad 68460  # fetch and decode live telemetry
python prototype/src/live.py <capture.parquet> # train and detect on a live capture
python prototype/src/populate_db.py --show 2   # one anomaly and its explanation
```

## Design decisions worth defending

**Detection is per channel, and so is the forecaster.**
A joint multivariate model was built first, on the reasoning that one model puts
every channel's error on a common footing. It does, and it still failed: scoring
the fleet by its loudest channel let one noisy channel raise the score
everywhere, and top-1 attribution came out at 0.09. The fix was to forecast and
threshold each channel independently, then put them on a common footing with
*robust z-scores* instead — median and MAD of that channel's own error — which
lifted attribution above 0.80 immediately. `joint.py` keeps the multivariate
model as a fallback when per-channel checkpoints are absent.

**Attribution reuses the errors the detector already computed.**
No SHAP, no permutation importance, no second forward pass. Per-channel
contribution comes from the same forecast residuals used for detection, so
explanation is effectively free and stays viable in real time. This was a
deliberate constraint from the problem brief.

**Presentation is per incident.**
Scoring is per channel because NASA's labels are, but an operator should not
receive 30 alarms for one fault. Co-firing channel events are grouped into a
single incident whose explanation lists the member channels by contribution
share, ordered by onset so the first mover is visible — which is not always the
loudest. One SMAP incident is driven 93% by a channel at 511σ that moved
*second*.

**Dynamic thresholds, not a hand-picked sigma.**
Each channel gets its own threshold by telemanom's nonparametric rule, followed
by their pruning step. A fixed sigma cut produced 399 detections against 37
labelled anomalies on SMAP — precision 0.17.

**Every explanatory signal must beat a random baseline.**
A signal that fires before 80% of faults is worthless if it fires before 80% of
quiet moments too. Anything claiming to explain or forewarn is calibrated
against matched random points and must clear a lift threshold before it is
reported at all — `factors.py` requires 2.0, `early.py` and `prewarning.py`
require 1.5. Three signals that looked explanatory died to this test: protocol
headers (present in every frame), commanding (asserted before 79% of all
readings), and early warning itself.

**Negative results are recorded, not buried.**
`reports/RESULTS.md` documents what does not work and why, including two ideas
that cost days: cross-channel Mahalanobis detection (zero gain under honest
selection) and early warning (no precursor exists in the data).

## Data

**NASA SMAP + MSL (Curiosity)** from the telemanom benchmark — 81 channels, 54
SMAP and 27 MSL, all of them monitored. The original S3 link in the telemanom
README is dead (403); `fetch_data.py` pulls from a maintained parquet mirror on
HuggingFace (`appleparan/telemanom`) plus NASA's `labeled_anomalies.csv`. NASA's
channel list contains P-2 twice, which `config.py` deduplicates — left in, it
double-counts detections. The joint model uses the 27 SMAP and 21 MSL channels
sharing a common timeline; per-channel detection uses all 81.

**ESA Anomaly Dataset** (`esa.py`, 3.7 GB, not in the repo) — 76 channels over
14 years with 1,203 classified anomalies, used to retest early warning on the
population most favourable to it. Unlike NASA it labels anomaly *class* and
separates a genuine fault from a rare nominal event.

**SatNOGS live captures** (`satnogs.py`) — real decoded telemetry from COSMO and
GRBBeta via the volunteer ground-station network. Unlabelled: nobody publishes
when a cubesat malfunctioned, so nothing here is scored. It demonstrates the
system on a real satellite; NASA's benchmark is what establishes that it works.

## Layout

| path | purpose |
|---|---|
| `src/config.py` | channel lists, hyperparameters, paths |
| `src/fetch_data.py` | dataset download |
| `src/data.py` | loading, alignment, sliding windows |
| `src/model.py` | LSTM forecaster + per-channel scaler |
| `src/train.py`, `src/train_channels.py` | joint and per-channel training |
| `src/detect.py` | thresholding, incident grouping, **attribution** |
| `src/novelty.py` | faults the forecaster has adapted to |
| `src/evaluate.py`, `src/tune.py` | scoring, cross-mission selection |
| `src/baseline_limits.py`, `src/baseline_cost.py` | the fixed-limit comparison |
| `src/early.py`, `src/prewarning.py`, `src/precursor.py` | early warning, measured |
| `src/factors.py` | external causes (commanding, space weather) |
| `src/sensors.py` | what each channel measures, and in what unit |
| `src/satnogs.py`, `src/live.py` | live ingestion, live training and detection |
| `src/esa.py`, `src/run_esa_precursor.py` | the ESA dataset and its precursor test |
| `src/sources.py` | one registry over benchmark and live sources |
| `src/db.py`, `src/populate_db.py` | the 7-table schema from the ER diagram |
| `dashboard/app.py` | Streamlit operator dashboard |
| `reports/RESULTS.md` | every measured number, including the negative ones |

## Results

`reports/RESULTS.md` carries the current numbers, how they compare to the
telemanom baseline, and the recorded negative results. `reports/evaluation.json`
holds the latest raw metrics.

## Tests

```bash
python -m pytest prototype/tests -q
```

55 regression tests. Each corresponds to a defect that actually occurred here,
written against the specific wrong behaviour rather than as generic coverage —
because the failures that mattered in this project were not crashes. They
produced plausible numbers that were wrong: a "60σ anomaly" in a network packet
header, a noise filter that never fired, anomalies scored as missed in data that
was never loaded, and a GPS channel reporting the satellite's own orbit as a
fault. Reading the code did not catch those; a test does.
