"""Early warning built as its own detector, not borrowed from the alert path.

Lowering the alert threshold does not produce early warning. It was measured:
the resulting signal fires before 61% of real anomalies and 69% of arbitrary
moments - a lift below 1, meaning it says nothing. A threshold on the *current*
error answers "is this reading unusual right now", which is the wrong question
for anticipating a fault.

Degradation has a different signature. A failing component drifts: each reading
is individually unremarkable, but the errors lean consistently one way, or their
spread widens, well before any single one is large enough to trip an alert.
Three detectors are built here for that signature specifically:

* **CUSUM** accumulates how far the error sits above its own baseline, resetting
  whenever it falls back. A long run of small positive deviations builds a large
  cumulative sum while no individual reading is anomalous - which is exactly the
  case a threshold misses. This is the classical change-point method and the
  main hope here.
* **Trend** fits a slope to the recent error and fires when it climbs steadily.
  Sensitive to gradual degradation, blind to a step change.
* **Volatility** watches for the error's spread widening against its own past,
  since instability often precedes failure even when the mean is unchanged.

Every one is scored the same way and against the same bar: it must fire before a
real anomaly meaningfully more often than before a random moment on the same
channel. Anything that does not is reported as not working. That test is what
caught the lowered-threshold approach, and it applies here unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from config import MIN_RUN, REPORT_DIR, SPACECRAFT
from evaluate import wilson_interval
from detect import rmt_tension

MAX_CREDIBLE_LEAD = 500        # readings; beyond this a "warning" is unrelated
MIN_USEFUL_LIFT = 1.5          # below this the detector is noise
RANDOM_TRIALS_PER_ANOMALY = 5


# ---------------------------------------------------------------- detectors --
def cusum_alarm(err: np.ndarray, slack_k: float = 1.5,
                threshold_h: float = 10.0, clip: float = 5.0) -> np.ndarray:
    med = np.median(err)
    mad = np.median(np.abs(err - med)) * 1.4826
    scale = max(mad, 1e-9)
    z = np.clip((err - med) / scale, -clip, clip)

    s = np.zeros_like(z)
    acc = 0.0
    for i, v in enumerate(z):
        acc = max(0.0, acc + v - slack_k)
        s[i] = acc
    return s > threshold_h


def trend_alarm(err: np.ndarray, window: int = 60,
                min_slope_sigma: float = 2.0) -> np.ndarray:
    """Fires where the error has been climbing steadily.

    The slope of a least-squares line over a trailing window, expressed in units
    of the error's own spread so channels remain comparable. A steady climb is
    the thing a point threshold cannot see: every reading in the run-up can sit
    comfortably below the alert level while the trajectory is unmistakable.
    """
    n = len(err)
    out = np.zeros(n, dtype=bool)
    if n <= window:
        return out
    med = np.median(err)
    mad = np.median(np.abs(err - med)) * 1.4826
    scale = max(mad, 1e-9)

    x = np.arange(window, dtype=np.float64)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    # Rolling slope via a strided view: one pass, no Python loop over windows.
    view = np.lib.stride_tricks.sliding_window_view(err, window)
    slopes = (view - view.mean(axis=1, keepdims=True)) @ x_c / denom
    # Slope per reading, scaled to the channel's own noise and window length.
    normalised = slopes * window / scale
    out[window - 1:] = normalised > min_slope_sigma
    return out


def volatility_alarm(err: np.ndarray, window: int = 60,
                     ratio: float = 2.5) -> np.ndarray:
    """Fires where the error's spread has widened against its own history."""
    n = len(err)
    out = np.zeros(n, dtype=bool)
    if n <= window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(err, window)
    local = view.std(axis=1)
    baseline = max(float(np.median(local)), 1e-9)
    out[window - 1:] = local > ratio * baseline
    return out


def signed_cusum_alarm(y_t: np.ndarray, y_p: np.ndarray, slack_k: float = 0.5,
                       threshold_h: float = 5.0, clip: float = 3.0) -> np.ndarray:
    raw_err = y_t - y_p
    med = np.median(raw_err)
    mad = np.median(np.abs(raw_err - med)) * 1.4826
    scale = max(mad, 1e-9)
    z = np.clip((raw_err - med) / scale, -clip, clip)
    
    s_pos = np.zeros_like(z)
    s_neg = np.zeros_like(z)
    acc_pos = 0.0
    acc_neg = 0.0
    for i, v in enumerate(z):
        acc_pos = max(0.0, acc_pos + v - slack_k)
        acc_neg = max(0.0, acc_neg - v - slack_k)
        s_pos[i] = acc_pos
        s_neg[i] = acc_neg
    return np.maximum(s_pos, s_neg) > threshold_h


def autocorr_alarm(err: np.ndarray, window: int = 60, threshold: float = 0.8) -> np.ndarray:
    """Fires when the rolling lag-1 autocorrelation exceeds the threshold."""
    n = len(err)
    out = np.zeros(n, dtype=bool)
    if n <= window:
        return out
    
    # Calculate rolling lag-1 autocorrelation
    view = np.lib.stride_tricks.sliding_window_view(err, window)
    v1 = view[:, :-1]
    v2 = view[:, 1:]
    
    v1_mean = v1.mean(axis=1, keepdims=True)
    v2_mean = v2.mean(axis=1, keepdims=True)
    
    v1_c = v1 - v1_mean
    v2_c = v2 - v2_mean
    
    num = (v1_c * v2_c).sum(axis=1)
    den = np.sqrt((v1_c**2).sum(axis=1) * (v2_c**2).sum(axis=1))
    
    den = np.where(den == 0, 1e-9, den)
    acf = num / den
    
    out[window - 1:] = acf > threshold
    return out


def skewness_alarm(err: np.ndarray, window: int = 60, threshold: float = 2.0) -> np.ndarray:
    """Fires when the rolling absolute skewness exceeds the threshold."""
    n = len(err)
    out = np.zeros(n, dtype=bool)
    if n <= window:
        return out
        
    view = np.lib.stride_tricks.sliding_window_view(err, window)
    mean = view.mean(axis=1, keepdims=True)
    std = view.std(axis=1)
    
    std = np.where(std == 0, 1e-9, std)
    
    m3 = ((view - mean)**3).mean(axis=1)
    skew = m3 / (std**3)
    
    out[window - 1:] = np.abs(skew) > threshold
    return out

DETECTORS = {
    "cusum": cusum_alarm,
    "trend": trend_alarm,
    "volatility": volatility_alarm,
    "signed_cusum": signed_cusum_alarm,
    "autocorr": autocorr_alarm,
    "skewness": skewness_alarm,
}

def _sustained(flag: np.ndarray, min_run: int) -> int | None:
    """First index where `flag` stays true for `min_run` readings."""
    run = 0
    for i, v in enumerate(flag):
        run = run + 1 if v else 0
        if run >= min_run:
            return i - run + 1
    return None

# ------------------------------------------------------------------ scoring --
def evaluate_detectors(result: dict, bundle, min_run: int = MIN_RUN,
                       max_lead: int = MAX_CREDIBLE_LEAD,
                       seed: int = 0) -> dict:
    """Lead time and random-baseline lift for every early-warning detector."""
    err = result["err"]
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    offset = int(result["t"][0])
    channels = result["channels"]
    scorable = bundle.scorable_labels(result.get("channel_lengths"))
    rng = np.random.default_rng(seed)

    hits = {k: [] for k in DETECTORS}
    rand = {k: [0, 0] for k in DETECTORS}

    for j, ch in enumerate(channels):
        col = err[:, j]
        yt = y_true[:, j]
        yp = y_pred[:, j]
        valid = np.flatnonzero(~np.isnan(col))
        if valid.size == 0:
            continue
        col = col[: valid[-1] + 1]
        yt = yt[: valid[-1] + 1]
        yp = yp[: valid[-1] + 1]

        # Each detector's alarm over the whole channel, computed once.
        alarms = {}
        for k, fn in DETECTORS.items():
            if k == "signed_cusum":
                alarms[k] = fn(yt, yp)
            else:
                alarms[k] = fn(col)

        starts = [w[0] - offset for w in scorable.get(ch, [])
                  if 0 < w[0] - offset < len(col)]
        for s0 in starts:
            seg = slice(max(0, s0 - max_lead), s0)
            for k, alarm in alarms.items():
                at = _sustained(alarm[seg], min_run)
                if at is not None:
                    hits[k].append(len(alarm[seg]) - at)

        lo_bound = max_lead + min_run + 1
        if not starts or len(col) <= lo_bound:
            continue

        # The baseline must be drawn from genuinely quiet stretches. A random
        # point can otherwise land inside or just after a labelled anomaly,
        # where the error is already elevated, so the "no fault here" baseline
        # is contaminated by faults - which pushes every lift below 1.0 and
        # makes even a working detector look worse than chance.
        blocked = np.zeros(len(col), dtype=bool)
        for w in scorable.get(ch, []):
            a = max(0, w[0] - offset - max_lead)
            b = min(len(col), w[1] - offset + max_lead)
            if b > a:
                blocked[a:b] = True

        candidates = np.flatnonzero(~blocked[lo_bound:]) + lo_bound
        if candidates.size == 0:
            continue
        picks = rng.choice(candidates,
                           size=min(len(candidates),
                                    len(starts) * RANDOM_TRIALS_PER_ANOMALY),
                           replace=False)
        for r0 in picks:
            seg = slice(int(r0) - max_lead, int(r0))
            for k, alarm in alarms.items():
                rand[k][1] += 1
                if _sustained(alarm[seg], min_run) is not None:
                    rand[k][0] += 1

    n_windows = sum(
        1 for ch in channels for w in scorable.get(ch, [])
        if 0 < w[0] - offset
    )

    out = {}
    for k in DETECTORS:
        warned = len(hits[k])
        rate = warned / n_windows if n_windows else 0.0
        base = rand[k][0] / rand[k][1] if rand[k][1] else 0.0
        
        if base > 0:
            lift = rate / base
        elif rate > 0:
            lift = float('inf')
        else:
            lift = None
            
        out[k] = {
            "warned": warned,
            "of": n_windows,
            "rate": round(rate, 4),
            "rate_95ci": wilson_interval(warned, n_windows),
            "fires_at_random": round(base, 4),
            "lift": None if lift is None else (round(lift, 3) if lift != float('inf') else float('inf')),
            "median_lead": int(np.median(hits[k])) if hits[k] else None,
            "mean_lead": round(float(np.mean(hits[k])), 1) if hits[k] else None,
            "is_early_warning": bool(lift is not None and lift >= MIN_USEFUL_LIFT),
        }
    return out


def evaluate_system_detectors(bundle, max_lead: int = MAX_CREDIBLE_LEAD, min_run: int = MIN_RUN, seed: int = 0) -> dict:
    """System-wide evaluation for RMT, based on unified anomalies across channels."""
    data = bundle.test
    T_len = data.shape[0]
    
    alarm = rmt_tension(data, window=60) > 2.0
    
    mask = np.zeros(T_len, dtype=bool)
    for ch, seqs in bundle.labels.items():
        for s, e in seqs:
            s, e = max(0, s), min(T_len - 1, e)
            if s <= e:
                mask[s:e+1] = True
                
    starts = []
    in_anomaly = False
    for i in range(T_len):
        if mask[i] and not in_anomaly:
            starts.append(i)
            in_anomaly = True
        elif not mask[i] and in_anomaly:
            in_anomaly = False
            
    hits = []
    for s0 in starts:
        seg = slice(max(0, s0 - max_lead), s0)
        at = _sustained(alarm[seg], min_run)
        if at is not None:
            hits.append(len(alarm[seg]) - at)
            
    rand_hits = [0, 0]
    lo_bound = max_lead + min_run + 1
    
    blocked = np.zeros(T_len, dtype=bool)
    for s0 in starts:
        a = max(0, s0 - max_lead)
        e0 = s0
        while e0 < T_len and mask[e0]: e0 += 1
        b = min(T_len, e0 + max_lead)
        if b > a: blocked[a:b] = True
        
    candidates = np.flatnonzero(~blocked[lo_bound:]) + lo_bound
    if candidates.size > 0:
        rng = np.random.default_rng(seed)
        picks = rng.choice(candidates, size=min(len(candidates), len(starts)*RANDOM_TRIALS_PER_ANOMALY), replace=False)
        for r0 in picks:
            seg = slice(int(r0) - max_lead, int(r0))
            rand_hits[1] += 1
            if _sustained(alarm[seg], min_run) is not None:
                rand_hits[0] += 1
                
    warned = len(hits)
    n_windows = len(starts)
    rate = warned / n_windows if n_windows else 0.0
    base = rand_hits[0] / rand_hits[1] if rand_hits[1] else 0.0
    
    if base > 0:
        lift = rate / base
    elif rate > 0:
        lift = float('inf')
    else:
        lift = None
    
    return {
        "warned": warned,
        "of": n_windows,
        "rate": round(rate, 4),
        "rate_95ci": wilson_interval(warned, n_windows),
        "fires_at_random": round(base, 4),
        "lift": None if lift is None else (round(lift, 3) if lift != float('inf') else "Infinity"),
        "median_lead": int(np.median(hits)) if hits else None,
        "is_early_warning": bool(lift is not None and lift >= MIN_USEFUL_LIFT),
    }


def report(name: str, verbose: bool = True) -> dict:
    from data import load_spacecraft
    from detect import detect

    bundle = load_spacecraft(name)
    res = evaluate_detectors(detect(bundle), bundle)
    rmt_res = evaluate_system_detectors(bundle)
    res["RMT_system"] = rmt_res

    if verbose:
        print(f"\n=== {name} (Per-Channel Detectors) ===")
        print(f"  {'detector':12s} {'warns':>7s} {'random':>7s} {'lift':>6s} "
              f"{'median lead':>12s}   verdict")
        per_ch_res = {k: v for k, v in res.items() if k != "RMT_system"}
        for k, v in sorted(per_ch_res.items(), key=lambda kv: -(kv[1]["lift"] or 0)):
            verdict = ("EARLY WARNING" if v["is_early_warning"]
                       else "no better than chance")
            lead = "-" if v["median_lead"] is None else str(v["median_lead"])
            print(f"  {k:12s} {v['rate']:6.0%} {v['fires_at_random']:6.0%} "
                  f"{str(v['lift']):>6s} {lead:>12s}   {verdict}")
        
        print(f"\n=== {name} (System-Wide RMT) ===")
        print(f"  System Incidents Evaluated: {rmt_res['of']}")
        verdict = "EARLY WARNING" if rmt_res["is_early_warning"] else "no better than chance"
        lead = "-" if rmt_res["median_lead"] is None else str(rmt_res["median_lead"])
        print(f"  {'RMT':12s} {rmt_res['rate']:6.0%} {rmt_res['fires_at_random']:6.0%} "
              f"{str(rmt_res['lift']):>6s} {lead:>12s}   {verdict}")
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", default="all", choices=["all", *SPACECRAFT])
    args = ap.parse_args()
    targets = list(SPACECRAFT) if args.spacecraft == "all" else [args.spacecraft]
    blob = {sc: report(sc) for sc in targets}
    (REPORT_DIR / "early_warning.json").write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {REPORT_DIR / 'early_warning.json'}")
