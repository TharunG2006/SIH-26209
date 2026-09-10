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


## Reporting the uncertainty, and a second precision convention

Three follow-ups, prompted by the three weakest points in the numbers above.

### Attribution now carries a confidence interval

The headline attribution figure rests on a handful of matched events, and a
bare "0.90" invites a reader to assume a precision the sample cannot support.
Every proportion is now reported with a 95% Wilson interval (Wilson rather than
the normal approximation, which misbehaves badly for small n near 1 and will
happily return bounds above 1):

| | top-1 attribution | 95% CI | events |
|---|---|---|---|
| SMAP | 0.900 | [0.60, 0.98] | 10 |
| MSL | 0.857 | [0.49, 0.97] | 7 |
| **pooled** | **0.882** | **[0.66, 0.97]** | **17** |

The pooled figure is the one to quote: same measurement, larger sample, so the
interval is tighter. It should be stated as **"88% (95% CI 66-97%, n=17)"** and
never as a bare percentage. The sample cannot be enlarged without more labelled
missions - it is bounded by how many detections coincide with a labelled window,
which is bounded in turn by recall.

### Why MSL precision is lower, diagnosed

MSL's 0.625 against SMAP's 0.809 is not random. Of MSL's false alarms,
inspection shows most are not spurious alerts at all:

* **D-15 (2151-2155)** misses its own channel's labelled window by 11 timesteps.
* **M-3, M-5, M-1** each fire while NASA has labelled anomalies on *other*
  channels at that moment - a real fault was underway and a neighbouring
  channel was flagged.
* Only **F-4** and **M-4** are genuinely isolated.

The strict convention counts flagging M-3 while NASA labelled M-6 during the
same fault as a false alarm, even though the engineer was correctly told the
spacecraft was misbehaving right then. That strictness is deliberate and stays -
it is what makes these numbers comparable to telemanom - but the gap it creates
is now measured rather than left implicit.

### Operational precision, reported alongside

| | strict (headline) | operational | incidents |
|---|---|---|---|
| SMAP | 0.809 | 1.000, 95% CI [0.72, 1.00] | 10/10 |
| MSL | 0.625 | 0.778, 95% CI [0.45, 0.94] | 7/9 |

Operational precision asks whether an incident coincided with any labelled
anomaly on any channel - whether the operator was alerted while something was
genuinely wrong. Every SMAP incident was. **This is a weaker claim and must
never be quoted as the precision**; it is reported so the distance between the
two conventions is visible.

### Recall: a third attempt, also rejected

Taking the union of what each threshold signal flags - catching a channel by
whichever view sees it, rather than committing every channel to one - was added
as a third option and offered to the cross-mission tuner. **It was not
selected**: both missions still chose the raw error alone. Recall stands at
0.432 (SMAP) and 0.385 (MSL).

That is now three approaches to recall that either did not help or did not
survive out-of-sample selection. The honest conclusion is unchanged: closing
this gap needs more labelled missions or a detector that does not rely on
forecast error alone, not further tuning of this one.


# Results on the full 82-channel dataset

Everything above was measured while the system monitored 48 of NASA's channels.
It now monitors all of them, and the numbers below supersede the earlier ones.

| | SMAP | MSL | telemanom |
|---|---|---|---|
| precision | **0.868** | 0.514 | 0.855 / 0.926 |
| recall | **0.603** | 0.472 | 0.855 / 0.694 |
| F1 | **0.712** | 0.492 | 0.855 / 0.794 |
| labelled windows | 68 | 36 | — |
| operational precision | 0.950 | 0.875 | not reported |
| attribution top-1 | **1.0** (n=19) | 0.8571 (n=7) | cannot report |

Pooled attribution: **25/26 = 0.962, 95% CI [0.81, 0.99]**.

**SMAP precision 0.868 now exceeds the published baseline's 0.855**, and recall
rose from 0.432 to 0.603 purely by monitoring the channels that were being
ignored. Nothing about the model or the thresholds changed - the gain came from
removing a self-imposed blind spot over 40% of the ground truth. MSL's precision
fell (0.514), which is the honest cost of scoring against 36 labelled windows
instead of 19.

## Early warning: measured, and it does not work

The stated goal includes warning before a fault. That is now measured rather
than assumed, and the answer is negative.

| | warns before | random baseline | lift | verdict |
|---|---|---|---|---|
| SMAP alert | never | — | — | no early warning |
| SMAP watch | 35% | 47% | 0.755 | **not early warning** |
| MSL alert | 11% | 14% | 0.769 | **not early warning** |
| MSL watch | 61% | 69% | 0.88 | **not early warning** |

Read the first pass of this on its own and it looks like a success: "warns
before 61% of MSL anomalies, median lead 472 readings". Calibrated against
matched random points on the same channel, that same signal fires before 69% of
arbitrary moments. Every lift is **below 1.0** - each level speaks slightly less
readily before a real anomaly than before nothing at all.

So the system detects faults as they begin and does not anticipate them. The
claim to make is *detection with an explanation*, never prediction - and not
earlier detection either, which is measured and refuted further down.

This is the third signal in this project that looked explanatory and was not:
protocol headers selected as telemetry because they appear in every frame,
command bits that precede 79% of all readings, and now a warning threshold
crossed most of the time. Each was caught by the same test - does it fire more
often when something is actually wrong - and that test is now built into the
code rather than remembered. `factors.py` requires a lift of 2.0 before
reporting a cause; `prewarning.py` requires 1.5 before calling something early
warning.


## Novelty detection: the first idea that improved recall on both missions

Four attempts at recall had failed, and all four asked the same question in a
different way: *is the forecast error large?* That question has a structural
blind spot. An LSTM with a 250-reading window learns to follow a sustained level
shift, so once a fault persists the model predicts the faulty values accurately,
the error collapses, and the detector goes quiet while the channel is plainly
broken.

Diagnosing the misses rather than guessing made it concrete. On MSL the forecast
error stays under 3 sigma for **46%** of everything missed; on SMAP the raw
values sit more than two training-sigma from where they ever operated for **29%**
of misses. The information was there - forecast error was the wrong question.

`novelty.py` asks whether a channel is *operating where it used to*, comparing a
rolling median of the test values against the training envelope. The model is
not involved, so it cannot be fooled by the model adapting.

### The threshold had to be per channel

The first version used one absolute shift threshold for every channel. Chosen on
SMAP it was far too loose for MSL, whose precision fell from 0.889 to 0.291
while recall tripled - a swap, not a gain. What counts as "far from normal"
depends on how much a channel drifts during healthy operation, and that differs
per channel and per spacecraft.

Each channel is now compared against a multiple of its own training-time wander,
measured by applying the same statistic to the training data. That removed the
mission dependence, and one setting now transfers:

| Held out | forecast only | + novelty | change |
|---|---|---|---|
| **SMAP** | P 1.000 R 0.382 F1 0.553 | P 0.893 **R 0.515** F1 **0.653** | recall **+0.132**, F1 +0.100 |
| **MSL** | P 0.889 R 0.222 F1 0.356 | P 0.789 **R 0.333** F1 **0.469** | recall **+0.111**, F1 +0.113 |

Settings were chosen on the *other* mission, as everywhere else here. Recall
improves on both, F1 improves on both, and precision gives up 11 points on SMAP
and 10 on MSL - a real trade rather than a swap.

### What did not work, recorded

A joint Mahalanobis detector over the residual vector, scoring cross-channel
combinations that never occur during healthy operation, was built on the
reasoning that a fault showing as five channels each moving two sigma is
invisible to a per-channel detector. On MSL it appeared to add three windows the
per-channel detector missed. Under cross-mission threshold selection it adds
**exactly zero** on both missions: the apparent gain came entirely from choosing
the quantile against MSL's own labels.

## Against a conventional fixed-limit alarm: we are not earlier, we are quieter

An earlier note in this project claimed the detector fires before a fixed-limit
alarm on 16 of 26 SMAP anomalies. That figure was never recorded here and does
not reproduce. Measured properly it is the opposite.

The baseline is the redline an operator would set knowing only healthy
behaviour: the minimum and maximum of the training series, applied unchanged to
the test series. A window counts as caught when a detection overlaps it, the
same rule the evaluator uses.

| | SMAP | MSL |
|---|---|---|
| both alarms caught it | 17 | 6 |
| ...we spoke first | **0** | **0** |
| ...the limit spoke first | 12 | 6 |
| ...same reading | 5 | 0 |
| only we caught it | 9 | 2 |
| only the limit caught it | 10 | 14 |
| neither | 32 | 14 |

We are never first. A limit set at the edge of the training range trips the
instant a reading leaves that range, whereas this detector waits for a
forecast error to persist for `min_run` readings and clear a threshold. Waiting
is the whole point, and it costs time.

What that waiting buys is the actual result:

| on healthy readings only | SMAP | MSL |
|---|---|---|
| fixed limit false alarms | 31,518 / 366,509 = **8.60%** | 11,299 / 59,213 = **19.08%** |
| ours | 142 / 366,509 = **0.04%** | 25 / 59,213 = **0.04%** |
| channels crying wolf | 14/54 vs 2/54 | 12/27 vs 3/27 |

The limit is first because it is always talking: on MSL it flags nearly a fifth
of all healthy telemetry. At **215x fewer false alarms on SMAP and 452x fewer on
MSL**, this detector still catches 9 SMAP and 2 MSL windows the limit never
catches at all, and names the responsible sensor in 25 of 26 matched events.

So the defensible claim is not earlier detection. It is *comparable coverage at a
false-alarm rate two orders of magnitude lower, with an explanation attached*.
Any wording in this repository promising earlier detection is wrong.

## Early warning, retested on ESA: still absent, and now on the fair population

The negative early-warning result above has an obvious objection: 59% of NASA's
anomalies are annotated `point`, meaning instantaneous. A method cannot warn
before something that has no before, so the failure might belong to the labels
rather than to the idea.

The ESA Anomaly Dataset answers that objection. Mission1 carries **1,203
anomalies, of which 1,142 are annotated `Subsequence`** - extended in time - and
only 61 `Point`. It also separates a genuine anomaly from a *rare nominal event*
(an uncommanded reset is a fault; the same reset after a telecommand is not),
which NASA's labels do not, so the events tested here are real faults. 76
channels, 14 years, 10.5 million readings per channel.

The test is `precursor.py`, which is deliberately **model-free**: detectors run
on a trailing-median residual that predicts nothing, so a negative cannot be
blamed on our forecaster. Baselines are drawn only from stretches with no
annotation of any category within 500 readings.

| Mission1, subsequence anomalies (n=1,127) | warns before | fires at random | lift |
|---|---|---|---|
| CUSUM | 88% | 88% | **0.99** |
| rolling trend | 68% | 71% | **0.96** |
| volatility | 11% | 11% | **1.04** |

| Mission1, point anomalies (n=61) | warns before | fires at random | lift |
|---|---|---|---|
| CUSUM | 100% | 100% | **1.00** |
| rolling trend | 30% | 27% | **1.08** |
| volatility | 0% | 0% | n/a |

"CUSUM warns before 88% of anomalies, and before 100% of point anomalies" is a
sentence one could put on a slide. It is worthless: the same detector fires
before 88% and 100% of arbitrary quiet moments. Every lift sits at 1.0.

This is the strongest form of the negative result available. It holds on a
second agency's missions, on 1,127 events rather than 68, on the anomaly class
most favourable to the claim, and without a model to blame. Prewarning is not
merely unachieved in this project - the precursor is not present in the
telemetry.

The four honest conclusions, in order of how hard they were to reach:

1. This detector names the responsible sensor in 25 of 26 matched events.
2. It does so at 0.04% false alarms against a fixed limit's 8.6% and 19.1%.
3. It is never earlier than that limit.
4. It cannot predict. Neither, on this evidence, can anything else.
