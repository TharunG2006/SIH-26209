"""Does a precursor exist in the raw telemetry, before any model sees it?

`early.py` asks whether our forecast error rises before a fault. That answer is
entangled with the forecaster: if the model is poor, a real precursor could be
missed. This module asks the more basic question - is there anything in the raw
signal at all - so a negative result cannot be blamed on the model.

Method is the same discipline used everywhere in this project. A detector fires
in the stretch before each labelled anomaly; the same detector is then run before
matched random points drawn from genuinely quiet telemetry on the same channel.
Lift is the ratio. A detector that fires just as readily before nothing is not a
warning, however impressive its raw hit rate looks.
"""
from __future__ import annotations

import numpy as np

from early import MIN_USEFUL_LIFT, cusum_alarm, trend_alarm, volatility_alarm
from evaluate import wilson_interval

DETECTORS = {"cusum": cusum_alarm, "trend": trend_alarm,
             "volatility": volatility_alarm}

MAX_LEAD = 500              # how far back a warning may credibly reach
RANDOM_TRIALS = 5           # matched quiet points per anomaly
MIN_RUN = 5                 # readings an alarm must hold to count


def residual(values: np.ndarray, window: int = 100) -> np.ndarray:
    """A model-free stand-in for forecast error: deviation from recent level.

    Subtracting a trailing median removes the channel's slow baseline without
    predicting anything, so what remains is the departure a persistence model
    would have failed to anticipate.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < window + 2:
        return np.zeros(n)
    pad = np.concatenate([np.full(window, v[0]), v])
    level = np.array([np.median(pad[i:i + window]) for i in range(n)])
    r = np.abs(v - level)
    scale = np.median(np.abs(r - np.median(r))) * 1.4826
    return r / max(scale, 1e-9)


def _sustained(flag: np.ndarray, min_run: int) -> int | None:
    """First index where `flag` stays true for `min_run` readings."""
    if flag.size < min_run:
        return None
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def test_channel(values, windows, rng, max_lead=MAX_LEAD):
    """Hits before real anomalies, and before matched quiet points."""
    r = residual(values)
    alarms = {k: fn(r) for k, fn in DETECTORS.items()}
    n = len(r)

    starts = [int(s) for (s, _e) in windows if 0 < int(s) < n]
    hits = {k: [] for k in DETECTORS}
    for s0 in starts:
        seg = slice(max(0, s0 - max_lead), s0)
        for k, a in alarms.items():
            at = _sustained(a[seg], MIN_RUN)
            if at is not None:
                hits[k].append(len(a[seg]) - at)

    blocked = np.zeros(n, dtype=bool)
    for (s, e) in windows:
        blocked[max(0, int(s) - max_lead):min(n, int(e) + max_lead)] = True
    lo = max_lead + MIN_RUN + 1
    rand = {k: [0, 0] for k in DETECTORS}
    if starts and n > lo:
        cand = np.flatnonzero(~blocked[lo:]) + lo
        if cand.size:
            picks = rng.choice(cand, size=min(cand.size,
                                              len(starts) * RANDOM_TRIALS),
                               replace=False)
            for r0 in picks:
                seg = slice(int(r0) - max_lead, int(r0))
                for k, a in alarms.items():
                    rand[k][1] += 1
                    if _sustained(a[seg], MIN_RUN) is not None:
                        rand[k][0] += 1
    return hits, rand, len(starts)


def summarise(per_channel) -> dict:
    """Pool per-channel counts into a lift table."""
    hits = {k: [] for k in DETECTORS}
    rand = {k: [0, 0] for k in DETECTORS}
    total = 0
    for h, r, n in per_channel:
        total += n
        for k in DETECTORS:
            hits[k].extend(h[k])
            rand[k][0] += r[k][0]
            rand[k][1] += r[k][1]

    out = {}
    for k in DETECTORS:
        warned = len(hits[k])
        rate = warned / total if total else 0.0
        base = rand[k][0] / rand[k][1] if rand[k][1] else 0.0
        lift = rate / base if base else None
        out[k] = {
            "warned": warned, "of": total, "rate": round(rate, 4),
            "rate_95ci": wilson_interval(warned, total),
            "fires_at_random": round(base, 4),
            "lift": None if lift is None else round(lift, 3),
            "median_lead": int(np.median(hits[k])) if hits[k] else None,
            "is_early_warning": bool(lift is not None
                                     and lift >= MIN_USEFUL_LIFT),
        }
    return out


def print_table(title: str, table: dict) -> None:
    print(f"\n{title}")
    print(f"  {'detector':11s} {'warns before':>13s} {'at random':>10s} "
          f"{'lift':>7s} {'median lead':>12s}  verdict")
    for k, v in table.items():
        lift = "n/a" if v["lift"] is None else f"{v['lift']:.2f}"
        lead = "-" if v["median_lead"] is None else str(v["median_lead"])
        verdict = "EARLY WARNING" if v["is_early_warning"] else "not early warning"
        print(f"  {k:11s} {v['warned']:5d}/{v['of']:<5d} {v['rate']:5.0%} "
              f"{v['fires_at_random']:9.0%} {lift:>7s} {lead:>12s}  {verdict}")
