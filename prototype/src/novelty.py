"""Detection against the training distribution, not against the forecast.

Forecast-error detection has a structural blind spot. An LSTM with a 250-reading
window learns to follow a sustained level shift: once a fault persists, the
model predicts the faulty values accurately and the error collapses back to
normal. The channel is plainly broken and the detector goes quiet.

Measured on the missed anomalies, this is not hypothetical. On MSL the forecast
error stays below 3 sigma for 46% of everything the per-channel detector misses,
and on SMAP the raw values sit more than two training-sigma away from where they
ever sat during healthy operation for 29% of the misses. The information is
there; forecast error is the wrong question to ask of it.

This asks a different one: *is the channel operating where it used to?* A rolling
median of the test values, expressed in units of the training spread, cannot be
fooled by the model adapting - the model is not involved. It complements rather
than replaces the forecaster, which remains far better at transients and at
contextual faults that stay inside the normal range.

Whether the combination helps is decided by cross-mission selection in
`evaluate_union`, never by choosing a threshold against the mission being
reported.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from config import MERGE_GAP, MIN_RUN, REPORT_DIR, SPACECRAFT

# Multiple of a channel's own natural wander, measured during training, before
# it counts as operating outside its learned envelope.
#
# An absolute sigma threshold does not transfer between missions: the value
# chosen on SMAP (3.0) collapsed MSL's precision from 0.889 to 0.291, because
# what counts as "far from normal" depends on how much a channel drifts during
# healthy operation, and that differs per channel and per spacecraft. Scaling
# each channel by its own training-time wander removes that dependence.
DEFAULT_MARGIN = 1.5

# Retained for callers that still pass an absolute threshold.
DEFAULT_SHIFT = 4.0

# Readings the shift must persist for. A level change is a slow thing; anything
# briefer is a transient the forecaster already handles better.
DEFAULT_WINDOW = 60


def training_envelope(train: np.ndarray) -> tuple[float, float]:
    """Robust centre and spread of healthy operation for one channel."""
    med = float(np.median(train))
    mad = float(np.median(np.abs(train - med))) * 1.4826
    if mad < 1e-9:
        # A channel that never moved in training: fall back to its full range,
        # so a constant channel that later varies is still detectable.
        span = float(train.max() - train.min())
        mad = span / 4.0 if span > 0 else 1.0
    return med, mad


def shift_score(test: np.ndarray, train: np.ndarray,
                window: int = DEFAULT_WINDOW) -> np.ndarray:
    """How far the channel is operating from where it was trained, over time.

    A rolling median is used rather than a mean so that a handful of spikes
    inside the window cannot manufacture a shift; what this is meant to catch is
    the level moving and staying moved.
    """
    med, mad = training_envelope(train)
    n = len(test)
    out = np.zeros(n, dtype=float)
    if n < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(test, window)
    rolling = np.median(view, axis=1)
    out[window - 1:] = np.abs(rolling - med) / mad
    return out


def natural_wander(train: np.ndarray, window: int = DEFAULT_WINDOW,
                   quantile: float = 0.99) -> float:
    """How far this channel drifts from its own centre during healthy operation.

    The same rolling-median statistic applied to the training data, which by
    definition contains no labelled fault. Its high quantile is the channel's
    own answer to "how far do I normally wander", which is what an absolute
    threshold was guessing at globally and getting wrong per mission.
    """
    score = shift_score(train, train, window)
    valid = score[window - 1:] if len(score) > window else score
    if valid.size == 0:
        return 1.0
    return max(float(np.quantile(valid, quantile)), 1e-6)


def channel_events(bundle, channels, offset: int, n_rows: int,
                   shift: float | None = None,
                   window: int = DEFAULT_WINDOW,
                   min_run: int = MIN_RUN,
                   merge_gap: int = MERGE_GAP,
                   margin: float = DEFAULT_MARGIN) -> list[dict]:
    """Per-channel detections from distributional shift alone.

    With `shift` given, that absolute threshold is used for every channel. Left
    as None, each channel is compared against `margin` times its own training
    wander - which is what lets one setting work across missions.
    """
    from detect import find_sequences

    events = []
    for ch in channels:
        train = bundle.full_train.get(ch)
        test = bundle.full_test.get(ch)
        if train is None or test is None or len(train) < window + 10:
            continue
        score = shift_score(test, train, window)
        thr = shift if shift is not None else margin * natural_wander(train, window)
        for s, e in find_sequences(score > thr, min_run, merge_gap):
            events.append({
                "channel": ch,
                "start": int(s), "end": int(e),
                "peak_shift": float(score[s:e + 1].max()),
                "threshold": float(thr),
            })
    return events


def evaluate_union(name: str, shift: float | None = None,
                   window: int = DEFAULT_WINDOW, verbose: bool = True,
                   margin: float = DEFAULT_MARGIN) -> dict:
    """Score the forecaster alone against the forecaster plus novelty."""
    from data import load_spacecraft
    from detect import detect
    from evaluate import _overlaps

    bundle = load_spacecraft(name)
    result = detect(bundle)
    offset = int(result["t"][0])
    scorable = bundle.scorable_labels(result.get("channel_lengths"))

    forecast_ev: dict[str, list] = {}
    for e in result["events"]:
        t = result["t"]
        forecast_ev.setdefault(e["channel"], []).append(
            (int(t[e["start"]]), int(t[e["end"]])))

    novel_ev: dict[str, list] = {}
    for e in channel_events(bundle, result["channels"], offset,
                            len(result["t"]), shift, window, margin=margin):
        novel_ev.setdefault(e["channel"], []).append((e["start"], e["end"]))

    def score(pred: dict) -> dict:
        tp = fp = 0
        hit = n_true = 0
        for ch in result["channels"]:
            truth = scorable.get(ch, [])
            p = pred.get(ch, [])
            n_true += len(truth)
            hit += sum(any(_overlaps(w, x) for x in p) for w in truth)
            tp += sum(any(_overlaps(x, w) for w in truth) for x in p)
            fp += sum(not any(_overlaps(x, w) for w in truth) for x in p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = hit / n_true if n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "n_detections": tp + fp}

    combined = {ch: forecast_ev.get(ch, []) + novel_ev.get(ch, [])
                for ch in result["channels"]}

    out = {
        "spacecraft": name,
        "forecast_only": score(forecast_ev),
        "novelty_only": score(novel_ev),
        "combined": score(combined),
        "shift": shift, "window": window, "margin": margin,
    }
    if verbose:
        print(f"\n=== {name} ===")
        for k in ("forecast_only", "novelty_only", "combined"):
            s = out[k]
            print(f"  {k:14s} P {s['precision']:.3f}  R {s['recall']:.3f}  "
                  f"F1 {s['f1']:.3f}  ({s['n_detections']} detections)")
        d = out["combined"]["recall"] - out["forecast_only"]["recall"]
        print(f"  recall change from adding novelty: {d:+.3f}")
    return out


def select_cross_mission(grid_margin=(1.0, 1.5, 2.0, 3.0, 5.0),
                         grid_window=(30, 60, 120)) -> dict:
    """Choose the novelty settings on one mission, report them on the other.

    Sweeping these against the mission being reported would be tuning on its
    test labels - the mistake this project has already made once with the
    threshold signal and once with the joint detector's quantile.
    """
    names = list(SPACECRAFT)
    report = {}
    for tune_on, report_on in ((names[1], names[0]), (names[0], names[1])):
        best, best_f1 = None, -1.0
        for mg in grid_margin:
            for win in grid_window:
                f1 = evaluate_union(tune_on, None, win, verbose=False,
                                    margin=mg)["combined"]["f1"]
                if f1 > best_f1:
                    best, best_f1 = (mg, win), f1
        held = evaluate_union(report_on, None, best[1], verbose=False,
                              margin=best[0])
        report[report_on] = {"chosen_on": tune_on,
                             "margin": best[0], "window": best[1], **held}
        f, c = held["forecast_only"], held["combined"]
        print(f"\nsettings chosen on {tune_on} (margin {best[0]}x the channel's "
              f"own wander, window {best[1]}) -> HELD-OUT {report_on}:")
        print(f"   forecast only : P {f['precision']:.3f}  R {f['recall']:.3f}  "
              f"F1 {f['f1']:.3f}")
        print(f"   + novelty     : P {c['precision']:.3f}  R {c['recall']:.3f}  "
              f"F1 {c['f1']:.3f}")
        print(f"   recall {c['recall'] - f['recall']:+.3f}   "
              f"F1 {c['f1'] - f['f1']:+.3f}")
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-mission", action="store_true",
                    help="select settings on the other mission (the honest test)")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = ap.parse_args()

    if args.cross_mission:
        blob = select_cross_mission()
    else:
        blob = {sc: evaluate_union(sc, None, args.window, margin=args.margin)
                for sc in SPACECRAFT}
    (REPORT_DIR / "novelty.json").write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {REPORT_DIR / 'novelty.json'}")
