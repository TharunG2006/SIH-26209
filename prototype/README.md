# Explainable Satellite Anomaly Detection — Prototype

SIH 2026, PS 26209. Working prototype for the AI-based satellite health
monitoring system described in `../CLAUDE.md`.

## What it does

Learns each spacecraft's normal telemetry behaviour with an LSTM, flags
timesteps where reality diverges from the forecast, and — the part that
distinguishes it from the NASA baseline — **explains every alert by naming the
channels and subsystems that caused it**, ranked by how much each contributed.

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
python prototype/src/train_channels.py      # 48 per-channel models, ~45 min
python prototype/src/populate_db.py         # fill the SQLite schema
python -m streamlit run prototype/dashboard/app.py
```

The two training steps are the slow part and only need doing once; everything
afterwards reads the saved checkpoints. Scripts can be run from any directory.

### The rest of the command line

```bash
python prototype/src/evaluate.py --sweep      # score against NASA's labels
python prototype/src/tune.py                  # cross-mission tuning, held-out results
python prototype/src/populate_db.py --alerts  # open alert queue
python prototype/src/populate_db.py --show 2  # one anomaly and its explanation
python prototype/src/satnogs.py --list        # live satellites with a decoder
python prototype/src/satnogs.py --norad 68460 # fetch and decode live telemetry
```

## Design decisions worth defending

**One multivariate model per spacecraft, not one per channel.**
NASA's telemanom trains a separate univariate model for each of the 82
channels. That makes per-channel errors incomparable — each comes from a
different model with a different error scale — so there is nothing to attribute
an anomaly *across*. We stack every channel of a spacecraft into one aligned
`(T, C)` matrix and train a single LSTM to forecast all of them jointly, which
puts every channel's error on a common footing. This is what makes the
explainability layer possible at all, and it is why the fault a real engineer
sees — *a pattern across several channels* — comes out as one ranked incident
rather than N disconnected alarms.

**Attribution reuses the errors the detector already computed.**
No SHAP, no permutation importance, no second forward pass. Per-channel
contribution is derived from the same forecast residuals used for detection, so
explanation is effectively free and stays viable in real time. This was a
deliberate constraint from the problem brief.

**Detection is per channel; presentation is per incident.**
NASA's labels are per channel, so scoring is per channel. But an operator
should not receive 30 alarms for one fault, so co-firing channel events are
grouped into a single incident whose explanation lists the member channels by
contribution share.

**Dynamic thresholds, not a hand-picked sigma.**
Each channel gets its own threshold chosen by telemanom's nonparametric rule,
followed by their pruning step. A fixed sigma cut was tried first and produced
399 detections against 37 labelled anomalies on SMAP — precision 0.17.

## Data

NASA SMAP + MSL (Curiosity) telemetry from the telemanom benchmark. The
original S3 link in the telemanom README is dead (403); `fetch_data.py` pulls
from a maintained parquet mirror on HuggingFace (`appleparan/telemanom`) and
NASA's `labeled_anomalies.csv` from the telemanom repo.

Of the 82 channels, we use the 27 SMAP and 21 MSL channels whose train/test
excerpts cover a common timeline, which is what allows joint modelling. Each
split is truncated to its shortest channel; all labelled anomaly windows fall
inside the retained range.

## Layout

| path | purpose |
|---|---|
| `src/config.py` | channel groups, hyperparameters, paths |
| `src/fetch_data.py` | dataset download |
| `src/data.py` | loading, alignment, sliding windows |
| `src/model.py` | multivariate LSTM forecaster + scaler |
| `src/train.py` | training with chronological validation split |
| `src/detect.py` | thresholding, incident grouping, **attribution** |
| `src/evaluate.py` | scoring vs NASA labels + telemanom comparison |
| `dashboard/app.py` | Streamlit operator dashboard |
| `reports/evaluation.json` | latest metrics |

## Results

See `reports/evaluation.json` for current numbers, and the notes in
`reports/RESULTS.md` for how they compare to the telemanom baseline and what
the remaining gaps are.
