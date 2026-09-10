# Explainable Satellite Anomaly Detection — SIH 2026 (PS 26209)

## Context
Team is competing in Smart India Hackathon 2026, Problem Statement 26209 (AICTE,
"Student Innovation — Space Technology", Category: Software). PS 26209 is an
open-ended slot — no fixed problem was given, so the team researched and chose
their own problem within the space-technology theme.

Team lead: Tharun. Idea-submission stage is done (PPT content drafted); this
repo is for the working prototype, per mentor's instruction to build one now.

## Chosen Problem
Satellites stream multivariate telemetry (power, temperature, sensors,
communication). Engineers currently monitor this manually across many
screens, which doesn't scale as constellations grow into the thousands.

Conventional fixed-limit alarms fail in two measured ways, and neither is the
one usually assumed. They do not fire *late*: a redline trips the instant a
reading leaves its range, which is earlier than any forecast-based detector can
manage. They fire *constantly* - on the NASA benchmark such a limit flags 8.6%
of healthy SMAP readings and 19.1% of healthy MSL readings, so real faults are
buried in noise operators have learned to ignore. And when one does fire it
reports only that a number left a band, never which sensor is responsible, by
how much, or what it affected next.

An operator facing a thousand spacecraft needs the opposite: an alarm that
speaks rarely and, when it does, names the sensor. That is the gap this project
targets - not earlier detection, and not prediction, both of which were measured
and are recorded as negative results in `prototype/reports/RESULTS.md`.

## Proposed Solution
An AI-based satellite health monitoring system:
1. One LSTM per channel learns that channel's normal behaviour and forecasts
   its next value. A single joint multivariate model was built first and is
   kept as a fallback; scoring the fleet by its loudest channel drove top-1
   attribution down to 0.09, which is what moved detection per-channel.
2. A large prediction-vs-actual deviation, expressed as a robust z-score
   against that channel's own error distribution, is an anomaly candidate.
   Per-channel scaling is what lets a quiet channel be heard at all.
3. **Explainability layer (the key differentiator):** for each flagged
   anomaly, show *which* channel(s) deviated most and by how much, and which
   moved first — not just a black-box score. Per-channel prediction-error
   attribution reuses the residuals detection already computed, so it costs no
   extra compute. Avoid SHAP — too slow for real-time telemetry. An
   attention-based variant remains a possible second explanation channel but
   is not built.
4. Co-firing channels are grouped into one incident, ranked by severity, and
   pushed to an operator dashboard as a single explained alert.

Measured and ruled out, so they should not be proposed again without new
evidence: early warning (no precursor exists - tested on 1,231 NASA and ESA
anomalies), commanding as an explanatory factor (lift 1.0), and cross-channel
Mahalanobis detection (zero gain under honest selection).

## Why this is defensible (competitive research already done)
- NASA has an open-source baseline: telemanom (LSTM, SMAP/MSL datasets,
  85.5% precision/recall on SMAP, 92.6%/69.4% on MSL). Real but imperfect —
  no stated real-time capability, runs on pre-split historical data.
  https://github.com/khundman/telemanom
- Aerospace Corp x Google Public Sector (2026): agentic AI for satellite
  anomaly resolution on Vertex AI — explicitly a proof-of-concept, not
  operational yet.
- Explainability for spacecraft telemetry is called out as still-unsolved
  in recent (2024) papers — most tools give only a raw anomaly score.
- Models trained on one mission's telemetry don't generalize well to
  another satellite's subsystems.
- Labeled failure data is scarce — most work reuses the same few public
  datasets (NASA SMAP/MSL, ESA anomaly dataset, OPS-SAT benchmark).
Full research + one-pager: see docs/Satellite_Anomaly_Detection_Brief.docx

## Data Sources
Training (labeled historical, has actual known anomalies):
- NASA telemanom dataset (SMAP + Mars Curiosity/MSL): https://github.com/khundman/telemanom
- ESA anomaly dataset (3 real ESA missions): https://github.com/esa/anomaly-dataset
- OPS-SAT benchmark: https://www.nature.com/articles/s41597-025-05035-3
- NASA Open Data Portal: https://data.nasa.gov/dataset/distributed-anomaly-detection-using-satellite-data-from-multiple-modalities

Live demo stream (real-time, unlabeled — for the "watch it work live" demo moment):
- SatNOGS Network (best option, documented API): https://satnogs.org/ · https://network.satnogs.org/ · API docs: https://docs.satnogs.org/projects/satnogs-db/en/latest/api.html
- NASA ISS live telemetry (via reverse-engineered feed, see ISS-Mimic): https://github.com/ISS-Mimic/Mimic
- TinyGS (community network): https://tinygs.com/

Strategy: train/validate on the labeled NASA/ESA data, demo live using SatNOGS.

## Architecture (see docs/diagrams.html for the full diagrams — Architecture,
## Runtime Workflow, Database Schema, Use Case)
Pipeline: [SatNOGS live feed / NASA-ESA historical data] -> Ingestion &
Normalization -> Feature Extraction (sliding-window vectors) -> LSTM Anomaly
Model -> Explainability Engine -> Alert Ranking & Manager -> writes to DB +
pushes to Operator Dashboard.

Database (7 tables): Satellite, Channel, TelemetryReading, Anomaly,
AnomalyContribution (links one Anomaly to multiple affected Channels —
this is what the explainability feature reads from), Alert, Operator.

## Suggested build order for the prototype
1. Load and visualize the SMAP/MSL labeled data
2. Train a baseline LSTM anomaly detector on it (can start from telemanom
   as reference, don't need to build from zero)
3. Add per-channel deviation/explainability scoring on top
4. Wrap it in a Streamlit dashboard: play back telemetry like a live
   stream, show a graph, pop up a clear alert with explanation on anomaly
5. Stretch goal: hook in live SatNOGS data for a real "live" demo moment

## Tech stack decided so far
Python, TensorFlow/PyTorch, Pandas/NumPy, Streamlit (dashboard),
PostgreSQL/SQLite (storage). No specialized hardware needed for the
prototype stage — cloud free tier / academic credits are enough.

## Folder layout
- docs/ — research brief (docx), architecture/workflow/ER/use-case diagrams (html)
- prototype/data/ — downloaded datasets (SMAP/MSL etc.)
- prototype/src/ — model training, inference, explainability code
- prototype/dashboard/ — Streamlit app
- prototype/notebooks/ — exploration notebooks
