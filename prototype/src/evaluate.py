"""Score the detector against NASA's labelled anomaly windows.

Uses the same event-level convention as the telemanom paper: a labelled
anomaly window counts as detected (true positive) if any predicted window
overlaps it; predicted windows that overlap no label are false positives.
This is what makes our numbers comparable to the published baseline
(SMAP 85.5% precision / 85.5% recall, MSL 92.6% / 69.4%).
"""
from __future__ import annotations

import argparse
import json

import sys

import numpy as np

from config import MIN_RUN, REPORT_DIR, SPACECRAFT, Z_MIN
from data import load_spacecraft
from detect import detect

# Published telemanom results, for side-by-side reporting only.
TELEMANOM = {
    "SMAP": {"precision": 0.855, "recall": 0.855, "f1": 0.855},
    "MSL": {"precision": 0.926, "recall": 0.694, "f1": 0.794},
}


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def event_scores(predicted: list[tuple[int, int]],
                 truth: list[tuple[int, int]]) -> dict:
    """Event-level precision / recall / F1 with overlap matching."""
    hit_truth = [any(_overlaps(p, t) for p in predicted) for t in truth]
    hit_pred = [any(_overlaps(p, t) for t in truth) for p in predicted]
    tp = sum(hit_pred)
    fp = len(predicted) - tp
    fn = len(truth) - sum(hit_truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = sum(hit_truth) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "n_predicted": len(predicted),
        "n_labelled": len(truth),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def attribution_accuracy(result: dict, bundle) -> dict:
    """Does the explanation name the right channel?

    For every detected event that overlaps a labelled window, check whether the
    channel our attribution ranks first (and top-3) is one of the channels NASA
    actually flagged in that window.  This measures the explainability layer
    itself, not just detection - and no published baseline reports it, because
    telemanom models each channel alone and so has nothing to attribute.
    """
    scorable = bundle.scorable_labels(result.get("channel_lengths"))
    truth_by_channel = {ch: scorable.get(ch, []) for ch in result["channels"]}
    top1 = top3 = matched = 0
    for a in result["anomalies"]:
        window = (a.start, a.end)
        truth_channels = {
            ch for ch, seqs in truth_by_channel.items()
            if any(_overlaps(window, s) for s in seqs)
        }
        if not truth_channels:
            continue
        matched += 1
        ranked = [c["channel"] for c in a.contributions]
        if ranked[:1] and ranked[0] in truth_channels:
            top1 += 1
        if any(c in truth_channels for c in ranked[:3]):
            top3 += 1
    return {
        "events_matched_to_labels": matched,
        "top1_channel_accuracy": round(top1 / matched, 4) if matched else None,
        "top3_channel_accuracy": round(top3 / matched, 4) if matched else None,
    }


def _in_range(w, rng) -> bool:
    """Keep a window if it starts inside the scoring range."""
    return rng is None or (rng[0] <= w[0] < rng[1])


def channel_event_scores(result: dict, bundle, time_range=None) -> dict:
    """Per-channel event scoring - the convention NASA's labels are written in.

    A predicted window only counts as a hit if it overlaps a labelled window
    *on the same channel*, so flagging the right moment on the wrong channel is
    correctly scored as a miss plus a false alarm.
    """
    predicted: dict[str, list[tuple[int, int]]] = {}
    for ev in result["events"]:
        t = result["t"]
        w = (int(t[ev["start"]]), int(t[ev["end"]]))
        if _in_range(w, time_range):
            predicted.setdefault(ev["channel"], []).append(w)

    scorable = bundle.scorable_labels(result.get("channel_lengths"))
    tp = fp = fn = n_truth = n_pred = 0
    for ch in result["channels"]:
        p = predicted.get(ch, [])
        truth = [w for w in scorable.get(ch, []) if _in_range(w, time_range)]
        n_pred += len(p)
        n_truth += len(truth)
        tp += sum(any(_overlaps(w, t) for t in truth) for w in p)
        fp += sum(not any(_overlaps(w, t) for t in truth) for w in p)
        fn += sum(not any(_overlaps(w, t) for w in p) for t in truth)

    hits = n_truth - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = hits / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "n_predicted": int(n_pred),
        "n_labelled": int(n_truth),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(name: str, z_min: float | None = None, min_run: int = MIN_RUN,
             per_channel: bool | None = None, time_range=None) -> dict:
    """Score one spacecraft.

    `time_range` restricts scoring to a (start, end) slice of the test series,
    which lets hyperparameters be chosen on an early stretch of a mission's own
    telemetry and reported on a later, untouched stretch - the temporal holdout
    a real operator would use.  Detection always runs over the whole series;
    only the scoring window is restricted.
    """
    bundle = load_spacecraft(name)
    result = detect(bundle, z_min=z_min, min_run=min_run,
                    per_channel=per_channel)

    scores = channel_event_scores(result, bundle, time_range=time_range)
    scores["incidents_shown_to_operator"] = len(result["anomalies"])
    scores.update(attribution_accuracy(result, bundle))
    # Padded cells are NaN once channels run at their own lengths, so the mean
    # has to skip them rather than propagate.
    scores["forecast_mae"] = float(
        np.nanmean(np.abs(result["y_true"] - result["y_pred"]))
    )
    scores["z_min"] = "dynamic" if z_min is None else z_min
    scores["min_run"] = min_run
    scores["telemanom_baseline"] = TELEMANOM.get(name)
    scores["forecaster"] = result["forecaster"]
    scores["labels_unreachable"] = bundle.dropped_label_count(
        result.get("channel_lengths"))
    return scores


def sweep(name: str, grid=(None, 2.5, 3.0, 4.0, 5.0, 6.0)) -> list[dict]:
    """Threshold sensitivity.

    `None` is the dynamic per-channel threshold the detector uses by
    default; the numeric entries are fixed sigma cuts, shown so the
    dynamic choice can be compared against hand-picked ones.
    """
    out = []
    for z in grid:
        s = evaluate(name, z_min=z)
        row = {k: s[k] for k in
               ("n_predicted", "precision", "recall", "f1",
                "top1_channel_accuracy")}
        row["z_min"] = "dynamic" if z is None else z
        out.append(row)
    return out


if __name__ == "__main__":
    # Windows consoles default to cp1252, which cannot render sigma.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", default="all", choices=["all", *SPACECRAFT])
    ap.add_argument("--sweep", action="store_true",
                    help="also report threshold sensitivity")
    ap.add_argument("--compare", action="store_true",
                    help="score both forecasters side by side")
    args = ap.parse_args()

    targets = list(SPACECRAFT) if args.spacecraft == "all" else [args.spacecraft]
    report = {}
    for sc in targets:
        s = evaluate(sc)
        report[sc] = s
        base = s["telemanom_baseline"] or {}
        print(f"\n=== {sc} ===")
        print(f"  labelled windows      {s['n_labelled']}")
        print(f"  channel detections    {s['n_predicted']}")
        print(f"  operator incidents    {s['incidents_shown_to_operator']}")
        print(f"  precision {s['precision']:.3f}   recall {s['recall']:.3f}"
              f"   F1 {s['f1']:.3f}")
        if base:
            print(f"  telemanom baseline    precision {base['precision']:.3f}"
                  f"   recall {base['recall']:.3f}   F1 {base['f1']:.3f}")
        print(f"  forecast MAE          {s['forecast_mae']:.4f}")
        print(f"  attribution top-1     {s['top1_channel_accuracy']}"
              f"   top-3 {s['top3_channel_accuracy']}"
              f"   (over {s['events_matched_to_labels']} matched events)")
        if args.sweep:
            print("  threshold sweep:")
            for row in sweep(sc):
                print(f"    z={str(row['z_min']):<8} n={row['n_predicted']:<4}"
                      f" P={row['precision']:.3f} R={row['recall']:.3f}"
                      f" F1={row['f1']:.3f} top1={row['top1_channel_accuracy']}")

        if args.compare:
            print("  forecaster comparison:")
            for flag, label in ((False, "joint multivariate"),
                                (True, "per-channel ensemble")):
                try:
                    c = evaluate(sc, per_channel=flag)
                except Exception as exc:            # ensemble may be untrained
                    print(f"    {label:22s} unavailable ({exc})")
                    continue
                print(f"    {label:22s} P={c['precision']:.3f} "
                      f"R={c['recall']:.3f} F1={c['f1']:.3f} "
                      f"MAE={c['forecast_mae']:.3f} "
                      f"top1={c['top1_channel_accuracy']}")

    (REPORT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT_DIR / 'evaluation.json'}")
