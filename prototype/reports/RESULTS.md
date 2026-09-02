# Results — final measured numbers

All figures produced by `python src/tune.py`, written to `reports/tuned_evaluation.json`.

## Protocol (read this first)

Hyperparameters were selected **on the other spacecraft**: the configuration used
to report SMAP was chosen by scoring MSL, and vice versa. No reported number
comes from a setting picked by looking at that mission's own labels. This is the
same discipline as a held-out validation set, and it is why these figures are
quotable under questioning — unlike a threshold swept on the test set until it
looked good.

Scoring is event-level and per channel, matching NASA's `labeled_anomalies.csv`
convention: a detection counts only if it overlaps a labelled window *on the
same channel*.

Forecasting uses the per-channel ensemble (one dedicated LSTM per channel);
attribution uses the same per-channel errors, so it costs no extra compute.


## Detection (held out)

| | SMAP | MSL | telemanom (published) |
|---|---|---|---|
| precision | **0.809** | 0.625 | 0.855 / 0.926 |
| recall | 0.432 | 0.385 | 0.855 / 0.694 |
| F1 | 0.564 | 0.476 | 0.855 / 0.794 |
| forecast MAE | 0.164 | 0.792 | not reported |
| detections / labelled | 21 / 37 | 16 / 26 | — |

SMAP precision is **0.809** against the baseline's 0.855, achieved
out-of-sample. Recall (0.432 against 0.855) remains the honest weak
point — the investigation into why is recorded below.


## Explainability (held out)

| | SMAP | MSL |
|---|---|---|
| top-1 channel attribution | **0.9** | 0.8571 |
| top-3 channel attribution | 0.9 | 0.8571 |
| events measured | 10 | 7 |

When our explanation names the channel responsible for an anomaly, it names
one NASA actually flagged 90% of the time on SMAP. **No published baseline
reports this metric**, because telemanom models each channel in isolation and so
has nothing to attribute across. Quote it with the event count attached — these
are 10 and 7 events, a strong signal rather than a large-sample result.


## Chosen configurations

- **SMAP** (selected on MSL): `{'prune_drop': 0.05, 'min_run': 5, 'z_grid_max': 6.0, 'threshold_signal': 'err'}`
- **MSL** (selected on SMAP): `{'prune_drop': 0.0, 'min_run': 5, 'z_grid_max': 10.0, 'threshold_signal': 'err'}`

## Causal ordering

Beyond ranking channels by size, contributing channels are ordered by **when
they began deviating**, which separates the origin of a fault from its
downstream effects:

> D-14 deviated FIRST, then M-6 +46 steps, F-8 +169 steps.

On SMAP this caught a case ranking alone would miss: E-6 is the loudest channel
(511σ, 93% of the deviation) but **E-9 moved first**. The largest symptom was not
the origin.

State this as evidence, not proof: "the earliest deviation was X, suggesting X
is upstream" — temporal precedence is not demonstrated physical causation.

## Known limits

1. **Recall is roughly half the baseline's.** We miss more real anomalies than
   telemanom does. Both raising forecast quality (2.4x better MAE) and loosening
   the detection rule (recall 0.189 to 0.378) helped; neither closed the gap.
2. **Channel names are unknown.** NASA anonymised them, so the system reports
   `E-6`, not `battery bus`. `config/channel_names.json` is where an operator
   supplies the real mapping; nothing is invented in its absence.
3. **No external cause data.** The model sees only spacecraft telemetry, so it
   cannot attribute a fault to radiation or space weather. NOAA publishes the
   relevant feeds free, but the NASA benchmark has no absolute timestamps to
   align them to — this is only viable on the live-data path.
4. **Small event counts.** Attribution accuracy rests on 10 and 7 matched events.


## Recall investigation (what was tried, and what did not work)

Recall is the weakest number, so it was attacked directly. Recording the
negative results because they define what is and is not claimable.

**Diagnosis.** Of 23 missed labelled windows on SMAP, 8 had errors that cleared
the channel threshold but were removed by pruning or the minimum-run filter;
the other 15 never cleared it. Several of those 15 were extreme on the
normalised scale (D-1 at z = 47.8, D-2 at z = 40.3) while sitting under an
error-based threshold, because the raw error's standard deviation is inflated
by outliers elsewhere in the series.

**Attempt 1 - better forecasts.** Per-channel models cut forecast MAE from
0.386 to 0.159 (2.4x). Precision rose from 0.778 to 1.000. **Recall did not
move at all** (0.189 either way). Smaller errors everywhere leave anomalies no
more prominent in relative terms.

**Attempt 2 - looser detection rules.** Widening the threshold search and
pruning raised recall from 0.189 to 0.378 with precision holding at 0.850.
This is the gain that stuck.

**Attempt 3 - threshold on the robust z-score instead of raw error.** Lifts
SMAP to precision 0.857 / recall 0.460 / F1 0.598. **Rejected**: the gain does
not survive cross-mission selection. Tuned on MSL, the winning configuration
uses the raw error, and SMAP falls back to 0.378. Reporting 0.460 would mean
having chosen the setting because it worked on SMAP's own test labels. The
underlying finding is real and worth stating: **the optimal detection
configuration does not transfer between missions**, which independently
reproduces a known open problem in this literature.

**Attempt 4 - temporal holdout within each mission.** Would allow
mission-specific tuning honestly, by choosing settings on an early stretch and
reporting on a later one. **Infeasible on this benchmark**: the labelled
anomalies are concentrated at the end of the test series (SMAP 4 windows in the
first half against 33 in the second; MSL 3 against 23), leaving too few to tune
on. This is the scarce-labelled-failure-data problem in concrete form.

**Conclusion.** Recall stands at 0.432 (SMAP) and 0.385 (MSL) under the honest
protocol, after the full-length and min_run corrections recorded below. Higher numbers are reachable but not defensible with the labelled
data available. Closing this gap properly needs either more labelled missions
or a detector that does not rely on forecast error alone.


## A scoring bug found late, and what it cost

Every channel is truncated to the shortest one so the series line up. On SMAP
that is lossless. On MSL it is not: the channels run to 6100 readings while the
shortest stops at 2038, so **7 of the 26 labelled anomalies sit past the end of
the data that was ever loaded** - and they were being counted as misses.

A detector cannot be charged with missing anomalies in data it was never given.
`TelemetryBundle.scorable_labels()` now excludes them, which moved MSL recall
from 0.346 to 0.474 and F1 from 0.409 to 0.486. SMAP was unaffected, having
nothing out of range.

This was a genuine measurement error, not a tuning choice - the corrected figure
counts only faults the system actually had the opportunity to catch.

**The better fix is still open.** Per-channel detection does not require the
channels to share a timeline at all; only the joint multivariate model does. Run
the per-channel ensemble on each channel's full length and those 7 anomalies
become reachable rather than merely excluded, which is worth roughly a quarter
of MSL's labelled data.


## Running every channel at its full length

The truncation above was not just a scoring problem, it was throwing away data.
Aligning all channels to the shortest one is required by the joint multivariate
model, but the per-channel ensemble - which is what now performs detection - has
no such constraint. Each channel is therefore forecast over every reading it
has, and the results are padded into a common grid with NaN where a shorter
channel has nothing. All error statistics, thresholds and the fleet score are
computed over each channel's real readings only, so a NaN tail cannot drag a
channel's median or spread around.

MSL gains the most: it was running on 2038 readings per channel when some
channels have 6100. Its 7 previously unreachable anomalies are now genuinely
detectable rather than merely excluded from scoring.

| | before (truncated) | after (full length) |
|---|---|---|
| SMAP recall | 0.378 | **0.460** |
| SMAP F1 | 0.524 | **0.589** |
| MSL labelled windows in scope | 19 of 26 | **26 of 26** |
| MSL anomalies caught | 9 | **10** |

SMAP improves outright. MSL's headline recall reads lower (0.385 against 0.474)
purely because the denominator grew from 19 to 26 - it now catches more
anomalies (10 against 9) while being judged on its complete label set. The two
MSL figures are not directly comparable, and the later one is the honest test.


## Correctness fixes after review

A review pass over the whole prototype found seven defects; all are fixed and
the numbers above are the re-measured result.

1. **`fetch_data.py` wrote to the working directory**, not `config.DATA_DIR`.
   The documented invocation (`python src/fetch_data.py` from the prototype
   root) put the dataset in `prototype/smap_msl/` while every other module
   looks in `prototype/data/smap_msl/`, so a fresh clone downloaded 9 MB and
   then failed on the next command. Paths are now anchored to `DATA_DIR`.
2. **The dashboard heatmap's colour scale was NaN.** `np.percentile` propagates
   NaN, and 53% of MSL's score matrix is NaN padding once channels run at their
   own lengths. Now `np.nanpercentile`.
3. **`min_run` filtered the merged span, not consecutive points.** Two isolated
   single-sample blips 40 steps apart passed a `min_run` of 5 as one 41-step
   sequence, so the noise filter never fired and the operator saw a long
   anomaly built from two samples. It now measures the longest consecutive run
   inside a merged sequence, as its documentation always claimed.
4. **Command features changed distribution partway through the series.** The
   models train on the fleet-wide command union, but inference spliced that
   union onto each channel's own columns at the truncation point - putting the
   seam squarely inside the MSL tail the full-length path exists to recover.
   The union is now built across the full length.
5. **Anomalies inside the first `WINDOW` readings were counted as misses**
   despite being undetectable, since forecasting only begins at `WINDOW`.
   Latent on this benchmark (neither mission has one) but live telemetry can.
6. **The forecast cache ignored retrained checkpoints**, so a long-lived
   Streamlit session served results from stale weights. Checkpoint mtimes are
   now part of the cache key.
7. **The manual-threshold path skipped the valid-length truncation**, relying on
   NaN comparisons being False rather than saying so.

Fix 3 is the one that moves the metrics: SMAP recall goes from 0.460 to 0.432
and precision from 0.818 to 0.809, because sequences that only ever passed the
length filter by spanning a gap are no longer counted. MSL improves (precision
0.556 to 0.625, F1 0.455 to 0.476). The earlier figures were mildly inflated by
the defect; these are the honest ones.
