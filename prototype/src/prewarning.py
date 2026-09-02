"""Early warning — does the system flag a fault before it is formally logged?

Detection answers "something is wrong now". An operator wants "something is
going wrong", with enough margin to act. The two are different claims and only
one of them has so far been measured.

NASA's labels make this testable rather than a matter of assertion. Each
labelled window has a start, and our score for that channel rises at some
reading. The gap between them is lead time:

    lead = labelled_start - first_reading_we_flagged

Positive means we flagged the channel before NASA's window opens - genuine
early warning. Negative means we lagged. Zero-ish means we detected it as it
happened, which is what the system was previously doing.

Two thresholds are involved. The alert threshold is the calibrated one the
reported metrics use; it is deliberately conservative. A *watch* threshold sits
below it and produces a lower-confidence "this channel is drifting" signal
earlier. Reporting both separates two honest claims - how early a confirmed
alert fires, and how early a soft warning could - instead of blurring them.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from config import MIN_RUN, REPORT_DIR, SPACECRAFT
from evaluate import wilson_interval

# A watch fires when a channel's smoothed error crosses this fraction of its
# alert threshold. It is not a detection and is never scored as one; it exists
# to quantify how much earlier a softer signal would have spoken.
WATCH_FRACTION = 0.5

# Only count a warning as belonging to a labelled anomaly if it precedes it by
# no more than this. A flag 3000 readings earlier is unrelated noise, not
# foresight, and counting it would manufacture a lead time that means nothing.
MAX_CREDIBLE_LEAD = 500


def _first_crossing(signal: np.ndarray, eps: float, upto: int,
                    min_run: int) -> int | None:
    """Earliest reading where `signal` sustains a breach of `eps` before `upto`."""
    if upto <= 0:
        return None
    flag = signal[:upto] > eps
    if not flag.any():
        return None
    run = 0
    for i, v in enumerate(flag):
        run = run + 1 if v else 0
        if run >= min_run:
            return i - run + 1
    return None


def lead_times(result: dict, bundle, min_run: int = MIN_RUN,
               watch_fraction: float = WATCH_FRACTION,
               max_lead: int = MAX_CREDIBLE_LEAD) -> dict:
    """Lead time per labelled anomaly, at both alert and watch thresholds."""
    import detect as D

    err, z = result["err"], result["z"]
    t = result["t"]
    offset = int(t[0])
    channels = result["channels"]
    scorable = bundle.scorable_labels(result.get("channel_lengths"))

    rows = []
    for j, ch in enumerate(channels):
        col = err[:, j]
        valid = np.flatnonzero(~np.isnan(col))
        if valid.size == 0:
            continue
        col = col[: valid[-1] + 1]
        eps = D.dynamic_threshold(col, min_run)
        watch_eps = eps * watch_fraction

        for start, end in scorable.get(ch, []):
            row_start = start - offset
            if row_start <= 0 or row_start >= len(col):
                continue
            # Look only at the run-up: a crossing after the window opens is
            # detection, not warning.
            window_lo = max(0, row_start - max_lead)
            segment = col[window_lo:row_start]
            alert_at = _first_crossing(segment, eps, len(segment), min_run)
            watch_at = _first_crossing(segment, watch_eps, len(segment), min_run)
            rows.append({
                "channel": ch,
                "labelled_start": int(start),
                "alert_lead": None if alert_at is None else int(len(segment) - alert_at),
                "watch_lead": None if watch_at is None else int(len(segment) - watch_at),
            })

    def summarise(key: str) -> dict:
        leads = [r[key] for r in rows if r[key] is not None]
        n = len(rows)
        if not leads:
            return {"warned": 0, "of": n, "rate": 0.0, "median_lead": None,
                    "mean_lead": None, "max_lead": None, "rate_95ci": (0.0, 0.0)}
        return {
            "warned": len(leads),
            "of": n,
            "rate": round(len(leads) / n, 4),
            "rate_95ci": wilson_interval(len(leads), n),
            "median_lead": int(np.median(leads)),
            "mean_lead": round(float(np.mean(leads)), 1),
            "max_lead": int(max(leads)),
        }

    return {
        "spacecraft": bundle.spacecraft,
        "labelled_windows_examined": len(rows),
        "alert": summarise("alert_lead"),
        "watch": summarise("watch_lead"),
        "watch_fraction": watch_fraction,
        "max_credible_lead": max_lead,
        "per_window": rows,
    }


def report(name: str, verbose: bool = True) -> dict:
    from data import load_spacecraft
    from detect import detect

    bundle = load_spacecraft(name)
    result = detect(bundle)
    out = lead_times(result, bundle)
    if verbose:
        print(f"\n=== {name} ===")
        print(f"  labelled anomalies examined: {out['labelled_windows_examined']}")
        for level in ("alert", "watch"):
            s = out[level]
            if s["median_lead"] is None:
                print(f"  {level:5s}: never fired before the window opened")
                continue
            lo, hi = s["rate_95ci"]
            print(f"  {level:5s}: warned before onset on {s['warned']}/{s['of']} "
                  f"= {s['rate']:.0%} (95% CI {lo:.0%}-{hi:.0%}), "
                  f"median lead {s['median_lead']} readings, max {s['max_lead']}")
        print("  A watch is a lower-confidence drift signal, not a detection, "
              "and is not counted in precision or recall.")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", default="all", choices=["all", *SPACECRAFT])
    args = ap.parse_args()
    targets = list(SPACECRAFT) if args.spacecraft == "all" else [args.spacecraft]
    blob = {}
    for sc in targets:
        r = report(sc)
        blob[sc] = {k: v for k, v in r.items() if k != "per_window"}
        blob[sc]["per_window"] = r["per_window"]
    (REPORT_DIR / "prewarning.json").write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {REPORT_DIR / 'prewarning.json'}")
