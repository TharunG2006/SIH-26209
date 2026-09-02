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
Faults often show up as a *pattern* across multiple channels, not a single
threshold breach, so simple rule-based alarms miss them or fire too late.

## Proposed Solution
An AI-based satellite health monitoring system:
1. LSTM (or Transformer) model learns each satellite's normal telemetry
   pattern and predicts expected values.
2. Large prediction-vs-actual deviation = anomaly candidate.
3. **Explainability layer (the key differentiator):** for each flagged
   anomaly, show *which* channel(s) deviated most and by how much — not
   just a black-box score. Use per-channel prediction-error attribution
   (native to the model, no extra compute) and/or attention-weight
   visualization if using an attention-based model. Avoid SHAP — too slow
   for real-time telemetry.
4. Alerts are ranked by severity and pushed to an operator dashboard.

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
