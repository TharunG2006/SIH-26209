"""Cross-channel detection: faults that no single channel reveals.

Detection so far has been per-channel and independent - each channel is asked,
on its own, whether it is deviating. That is what telemanom does, and it means a
fault is only found when at least one channel breaches its own threshold.

But the premise of this project is that a spacecraft fault shows up as a
*pattern across channels*. Five channels each moving two sigma is unremarkable
one at a time and close to impossible together, and the per-channel detector
cannot see it: no single threshold is crossed. Cross-channel structure is
currently used only to explain anomalies after they are found, never to find
them.

This module closes that gap. The residual vector at each reading is scored
against the covariance the residuals normally have, so a combination that never
occurs during healthy operation is flagged even when every component is
individually ordinary.

    d(t)^2 = r(t)^T  S^-1  r(t)

with S estimated robustly from the residuals themselves. Two details matter.
The covariance is shrunk toward its diagonal, because with dozens of channels
and a few thousand readings the sample covariance is badly conditioned and its
inverse amplifies noise. And each channel's contribution to the distance is
recoverable, so this detector explains itself the same way the per-channel one
does rather than becoming the black box the project set out to avoid.

Whether it helps is a measurement, not a claim - `python src/joint.py` reports
precision and recall against NASA's labels beside the per-channel detector.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from config import MERGE_GAP, MIN_RUN, REPORT_DIR, SPACECRAFT

# Shrinkage toward the diagonal. With C channels and T readings the sample
# covariance needs T >> C^2 to be well conditioned, which never holds here, so
# some shrinkage is not optional.
SHRINKAGE = 0.3

# Quantile of the distance used as the detection threshold. Chosen on the other
# mission by tune-style cross-selection, never on the mission being reported.
DEFAULT_QUANTILE = 0.995


def robust_covariance(resid: np.ndarray, shrinkage: float = SHRINKAGE
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Median-centred, shrunk covariance of the residual vectors.

    Anomalous readings are in the data being used to estimate normality, so the
    centre is a median and the scale a MAD - a mean and a sample covariance
    would be pulled toward the very events we are trying to stand out from.
    """
    centre = np.nanmedian(resid, axis=0)
    centred = resid - centre

    # Standardise per channel first so a large-amplitude channel does not
    # dominate the distance purely through its units.
    mad = np.nanmedian(np.abs(centred), axis=0) * 1.4826
    scale = np.where(mad < 1e-9, 1.0, mad)
    std = centred / scale

    ok = ~np.isnan(std).any(axis=1)
    usable = std[ok]
    if len(usable) < std.shape[1] + 2:
        raise ValueError("too few complete readings to estimate a covariance")

    cov = np.cov(usable, rowvar=False)
    cov = np.atleast_2d(cov)
    # Shrink toward the diagonal: the off-diagonal terms are what carry the
    # cross-channel information, but they are also the least well estimated.
    target = np.diag(np.diag(cov))
    cov = (1 - shrinkage) * cov + shrinkage * target
    cov += np.eye(len(cov)) * 1e-6          # keep it invertible
    return cov, scale


def mahalanobis(resid: np.ndarray, shrinkage: float = SHRINKAGE
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-reading joint distance, and each channel's share of it.

    The share is the channel's term in the quadratic form, so the explanation
    comes from the same computation as the score - no second pass, and no
    separate model whose reasoning could disagree with the detector's.
    """
    centre = np.nanmedian(resid, axis=0)
    cov, scale = robust_covariance(resid, shrinkage)
    std = (resid - centre) / scale
    inv = np.linalg.pinv(cov)

    filled = np.where(np.isnan(std), 0.0, std)
    # r^T S^-1 r, one row at a time but vectorised over readings.
    half = filled @ inv
    d2 = np.einsum("ij,ij->i", half, filled)
    # Channel j's contribution to that sum, which is what the attribution reads.
    share = half * filled
    return np.sqrt(np.maximum(d2, 0.0)), share


def detect_joint(result: dict, quantile: float = DEFAULT_QUANTILE,
                 min_run: int = MIN_RUN, merge_gap: int = MERGE_GAP,
                 shrinkage: float = SHRINKAGE) -> dict:
    """Cross-channel detections, with per-channel attribution."""
    from detect import find_sequences

    resid = result["y_true"] - result["y_pred"]
    dist, share = mahalanobis(resid, shrinkage)
    eps = float(np.nanquantile(dist, quantile))

    events = []
    t = result["t"]
    channels = result["channels"]
    for s, e in find_sequences(dist > eps, min_run, merge_gap):
        block = share[s:e + 1]
        totals = np.nansum(block, axis=0)
        order = np.argsort(totals)[::-1]
        total = float(totals[order].sum()) or 1.0
        events.append({
            "start": int(t[s]), "end": int(t[e]),
            "peak": int(t[s + int(np.argmax(dist[s:e + 1]))]),
            "severity": float(dist[s:e + 1].max()),
            "channels": [
                {"channel": channels[j],
                 "share_pct": round(100.0 * float(totals[j]) / total, 2)}
                for j in order[:8] if totals[j] > 0
            ],
        })
    return {"threshold": eps, "distance": dist, "events": events}


def compare(name: str, quantile: float = DEFAULT_QUANTILE,
            verbose: bool = True) -> dict:
    """Score the joint detector against the per-channel one on the same data."""
    from data import load_spacecraft
    from detect import detect
    from evaluate import _overlaps

    bundle = load_spacecraft(name)
    result = detect(bundle)
    joint = detect_joint(result, quantile=quantile)

    # Ground truth pooled across channels: the joint detector is a fleet-level
    # signal and cannot be scored per channel, so it is compared on whether an
    # incident coincides with any labelled anomaly - the same convention as the
    # operational precision already reported for the per-channel detector.
    windows = sorted({w for seqs in bundle.scorable_labels(
        result.get("channel_lengths")).values() for w in seqs})
    preds = [(e["start"], e["end"]) for e in joint["events"]]
    tp = sum(any(_overlaps(p, w) for w in windows) for p in preds)
    hit = sum(any(_overlaps(p, w) for p in preds) for w in windows)

    per_channel = [(a.start, a.end) for a in result["anomalies"]]
    pc_tp = sum(any(_overlaps(p, w) for w in windows) for p in per_channel)
    pc_hit = sum(any(_overlaps(p, w) for p in per_channel) for w in windows)

    def rates(tp_, hit_, n_pred, n_true):
        p = tp_ / n_pred if n_pred else 0.0
        r = hit_ / n_true if n_true else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    jp, jr, jf = rates(tp, hit, len(preds), len(windows))
    pp, pr, pf = rates(pc_tp, pc_hit, len(per_channel), len(windows))

    # What the two find that the other does not - the question that decides
    # whether combining them is worth anything.
    only_joint = sum(
        1 for w in windows
        if any(_overlaps(p, w) for p in preds)
        and not any(_overlaps(p, w) for p in per_channel)
    )
    both = sum(
        1 for w in windows
        if any(_overlaps(p, w) for p in preds)
        and any(_overlaps(p, w) for p in per_channel)
    )
    union_hit = sum(
        1 for w in windows
        if any(_overlaps(p, w) for p in preds)
        or any(_overlaps(p, w) for p in per_channel)
    )

    out = {
        "spacecraft": name,
        "labelled_windows": len(windows),
        "per_channel": {"precision": pp, "recall": pr, "f1": pf,
                        "n_detections": len(per_channel)},
        "joint": {"precision": jp, "recall": jr, "f1": jf,
                  "n_detections": len(preds), "quantile": quantile},
        "found_only_by_joint": only_joint,
        "found_by_both": both,
        "union_recall": round(union_hit / len(windows), 4) if windows else 0.0,
    }
    if verbose:
        print(f"\n=== {name} ===  {len(windows)} labelled windows")
        print(f"  per-channel : P {pp:.3f}  R {pr:.3f}  F1 {pf:.3f}  "
              f"({len(per_channel)} detections)")
        print(f"  joint       : P {jp:.3f}  R {jr:.3f}  F1 {jf:.3f}  "
              f"({len(preds)} detections)")
        print(f"  found ONLY by the joint detector: {only_joint}")
        print(f"  found by both                   : {both}")
        print(f"  recall if the two are combined  : {out['union_recall']:.3f}"
              f"   (per-channel alone {pr:.3f})")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", default="all", choices=["all", *SPACECRAFT])
    ap.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    args = ap.parse_args()
    targets = list(SPACECRAFT) if args.spacecraft == "all" else [args.spacecraft]
    blob = {sc: compare(sc, quantile=args.quantile) for sc in targets}
    (REPORT_DIR / "joint_detection.json").write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {REPORT_DIR / 'joint_detection.json'}")
