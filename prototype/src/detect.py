"""Turn forecast errors into ranked, explained anomalies.

Pipeline per spacecraft:
    forecast -> per-channel error -> EWMA smoothing -> robust z-score
    -> per-timestep fleet score -> contiguous anomaly sequences
    -> per-channel attribution (the explainability layer)
"""
from __future__ import annotations

from dataclasses import dataclass

import sys

import pathlib

import numpy as np
import torch

from config import (MERGE_GAP, MIN_RUN, MODEL_DIR, SMOOTH_WINDOW,
                    SUBSYSTEM, WINDOW, Z_MIN, channel_label)
from config import PRUNE_DROP as _DEFAULT_PRUNE_DROP

PRUNE_DROP = _DEFAULT_PRUNE_DROP
from data import TelemetryBundle, make_windows
from model import ChannelScaler, TelemetryForecaster
from train_channels import CH_DROPOUT, CH_HIDDEN, CH_LAYERS

SIGMA = "σ"

# Detection knobs, kept as module globals so tune.py can vary them without
# re-importing.  Defaults mirror config.py.
Z_SEARCH_MAX = 10.5

# Which signal the dynamic threshold is chosen on: the raw smoothed error, or
# its robust z-score.  "z" avoids the raw error's variance being inflated by
# outliers elsewhere in the series and lifts SMAP recall from 0.378 to 0.460 -
# but that gain does NOT survive cross-mission selection (MSL's own best config
# picks "err"), so shipping "z" as the default would amount to tuning on SMAP's
# test labels.  Default stays "err"; tune.py selects per mission and records
# the choice in reports/tuned_evaluation.json.
THRESHOLD_SIGNAL = "err"

# Forecasting is by far the expensive step and does not depend on any of the
# detection knobs, so results are memoised per spacecraft.  Without this, a
# hyperparameter sweep would re-run every channel model on every trial.
_FORECAST_CACHE: dict[str, tuple] = {}


# Forecasting a spacecraft means running every channel model over every
# reading, which takes minutes. Nothing about it changes between processes, so
# the result is also cached on disk - otherwise the dashboard pays that cost on
# every start, which is unusable for a live demo.
_DISK_CACHE = MODEL_DIR / "forecast_cache"


def clear_forecast_cache(disk: bool = False) -> None:
    _FORECAST_CACHE.clear()
    if disk and _DISK_CACHE.is_dir():
        for f in _DISK_CACHE.glob("*.npz"):
            f.unlink()


def _disk_path(spacecraft: str, per_channel) -> "pathlib.Path":
    return _DISK_CACHE / f"{spacecraft.lower()}_{per_channel}.npz"


def _load_disk(spacecraft: str, per_channel, fingerprint):
    """Cached forecast, if it was produced by exactly these checkpoints."""
    path = _disk_path(spacecraft, per_channel)
    if not path.exists():
        return None
    try:
        blob = np.load(path, allow_pickle=True)
        if tuple(blob["fingerprint"]) != tuple(fingerprint):
            return None          # models were retrained; recompute
        return (list(blob["channels"]), blob["y_true"], blob["y_pred"],
                blob["t"], str(blob["forecaster"]),
                dict(blob["lengths"].item()))
    except Exception:
        return None              # a corrupt cache must never break a run


def _save_disk(spacecraft: str, per_channel, fingerprint, payload) -> None:
    channels, y_true, y_pred, t, forecaster, lengths = payload
    _DISK_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(
            _disk_path(spacecraft, per_channel),
            fingerprint=np.array(fingerprint, dtype=object),
            channels=np.array(channels, dtype=object),
            y_true=y_true, y_pred=y_pred, t=t,
            forecaster=forecaster, lengths=np.array(lengths, dtype=object),
        )
    except Exception as exc:
        # Caching is an optimisation and must never break a run - but silently
        # discarding the error meant the cache appeared to work while writing
        # nothing, and the dashboard paid the full cost on every start.
        print(f"[detect] could not write forecast cache: "
              f"{type(exc).__name__}: {exc}", flush=True)


def _model_fingerprint(spacecraft: str) -> tuple:
    """Modification times of every checkpoint a forecast depends on.

    Included in the cache key so that retraining during a long-lived process -
    a Streamlit session, a notebook - invalidates the memoised forecasts instead
    of silently serving results from the old weights.
    """
    paths = [MODEL_DIR / f"{spacecraft.lower()}_lstm.pt"]
    d = MODEL_DIR / f"{spacecraft.lower()}_channels"
    if d.is_dir():
        paths.extend(sorted(d.glob("*.pt")))
    return tuple(p.stat().st_mtime_ns for p in paths if p.exists())


@dataclass
class Anomaly:
    """One detected event, with its per-channel explanation."""

    start: int                       # index into the test series
    end: int
    peak: int                        # timestep of maximum severity
    severity: float                  # max fleet z-score inside the window
    contributions: list[dict]        # ranked per-channel attribution
    subsystems: list[dict]           # same, rolled up by subsystem group

    @property
    def duration(self) -> int:
        return self.end - self.start + 1

    def top_channel(self) -> str:
        return self.contributions[0]["channel"] if self.contributions else "?"

    def chain(self) -> list[dict]:
        """Contributing channels ordered by when they began deviating."""
        return sorted(self.contributions, key=lambda c: c.get("chain_rank", 0))

    def first_mover(self) -> dict | None:
        """The channel that deviated first - the best candidate for the origin."""
        ch = self.chain()
        return ch[0] if ch else None

    def propagation(self) -> str:
        """Plain-language causal ordering, e.g. 'T-2 led, P-1 followed 40 steps later'."""
        ch = self.chain()
        if not ch:
            return "No channels to order."
        if len(ch) == 1:
            return f"{ch[0]['label']} deviated alone - no propagation to other channels."
        lead = ch[0]
        rest = [c for c in ch[1:] if c["lag"] > lead["lag"]]
        if not rest:
            return (f"{lead['label']} and {len(ch) - 1} other channel(s) deviated "
                    "simultaneously - no lead/lag separation.")
        parts = [f"{c['label']} +{c['lag'] - lead['lag']} steps" for c in rest[:3]]
        return (f"{lead['label']} deviated FIRST, then " + ", ".join(parts) +
                ". The earliest channel is the likelier origin; the rest may be "
                "downstream effects.")

    def explanation(self) -> str:
        """One-line, operator-readable cause statement."""
        if not self.contributions:
            return "No channel attribution available."
        parts = [
            "{} ({:.0f}%, {}, {:.1f}{})".format(
                c["channel"], c["share_pct"], c["direction"], c["z"], SIGMA
            )
            for c in self.contributions[:3]
        ]
        lead = self.subsystems[0]["subsystem"] if self.subsystems else "unknown"
        return "Driven by {}: {}. Peak severity {:.1f}{} at t={}.".format(
            lead, ", ".join(parts), self.severity, SIGMA, self.peak
        )


def load_channel_detectors(spacecraft: str):
    """Load the per-channel forecasters, if they have been trained."""
    d = MODEL_DIR / f"{spacecraft.lower()}_channels"
    if not d.is_dir():
        return None
    nets, scalers, windows = {}, {}, {}
    for f in sorted(d.glob("*.pt")):
        ck = torch.load(f, weights_only=False)
        net = TelemetryForecaster(1, n_cmd=ck.get("n_cmd", 24),
                                  hidden=ck.get("hidden", CH_HIDDEN),
                                  layers=CH_LAYERS, dropout=CH_DROPOUT)
        net.load_state_dict(ck["model"])
        net.eval()
        nets[ck["channel"]] = net
        scalers[ck["channel"]] = ChannelScaler().load_state_dict(ck["scaler"])
        # Inference must use the window the model was trained with, so it is
        # read from the checkpoint rather than assumed to be the global.
        windows[ck["channel"]] = ck.get("window", WINDOW)
    return (nets, scalers, windows) if nets else None


def forecast_per_channel(nets, scalers, channels, bundle, batch=256,
                         windows=None):
    """Forecast each channel with its own dedicated model, at its own length.

    Only the joint multivariate model needs the channels to share a timeline;
    a per-channel model does not.  Each channel is therefore run over every
    reading it has, and the results are padded into a common (T, C) grid with
    NaN where a shorter channel has no data.  Row i is timestep WINDOW + i for
    every channel, since each channel's own series starts at its own zero.
    """
    windows = windows or {}
    win = {ch: windows.get(ch, WINDOW) for ch in channels}
    lengths = {ch: len(bundle.full_test[ch]) for ch in channels}
    # Rows are offset by the largest window so every channel's row i refers to
    # the same reading index regardless of how much history its model needs.
    base = max(win.values())
    n = max(lengths.values()) - base
    t = np.arange(n) + base
    y_true = np.full((n, len(channels)), np.nan, dtype=np.float32)
    y_pred = np.full((n, len(channels)), np.nan, dtype=np.float32)

    for j, ch in enumerate(channels):
        values = bundle.full_test[ch]
        cmds = bundle.full_cmd[ch]
        w = win[ch]
        rows = len(values) - base
        if rows <= 0:
            continue
        scaler = scalers[ch]
        scaled = scaler.transform(values.reshape(-1, 1)).ravel()
        feats = np.ascontiguousarray(
            np.concatenate([scaled.reshape(-1, 1), cmds], axis=1),
            dtype=np.float32,
        )
        # A strided view costs nothing; materialising every window up front
        # would allocate ~200 MB per channel, so only each batch is copied.
        # Start far enough in that the first row lines up with `base`.
        offset = base - w
        view = np.lib.stride_tricks.sliding_window_view(
            feats, w, axis=0
        )[offset:offset + rows].transpose(0, 2, 1)
        preds = []
        with torch.no_grad():
            for i in range(0, rows, batch):
                chunk = np.ascontiguousarray(view[i:i + batch])
                preds.append(nets[ch](torch.from_numpy(chunk)).numpy())
        y_true[:rows, j] = values[base:base + rows]
        y_pred[:rows, j] = scaler.inverse(np.concatenate(preds)).ravel()
    return y_true, y_pred, t, lengths


def load_detector(spacecraft: str):
    ckpt = torch.load(MODEL_DIR / f"{spacecraft.lower()}_lstm.pt", weights_only=False)
    net = TelemetryForecaster(len(ckpt["channels"]))
    net.load_state_dict(ckpt["model"])
    net.eval()
    scaler = ChannelScaler().load_state_dict(ckpt["scaler"])
    return net, scaler, ckpt["channels"]


def forecast(net, scaler, values: np.ndarray, cmds: np.ndarray,
             batch: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the forecaster over a series.

    Returns (y_true, y_pred, t) in the original value units, where t is the
    index in `values` each row corresponds to.  The first WINDOW steps have no
    prediction and are excluded.
    """
    X, y, t = make_windows(scaler.transform(values), cmds, WINDOW)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            # X is a strided view, so each batch is copied here rather than the
            # whole array being materialised up front.
            chunk = np.ascontiguousarray(X[i:i + batch], dtype=np.float32)
            preds.append(net(torch.from_numpy(chunk)).numpy())
    y_pred = scaler.inverse(np.concatenate(preds))
    y_true = scaler.inverse(y)
    return y_true, y_pred, t


def _ewma(a: np.ndarray, span: int) -> np.ndarray:
    """Column-wise exponentially weighted moving average."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def error_scores(y_true: np.ndarray, y_pred: np.ndarray,
                 smooth: int = SMOOTH_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel robust z-scores of the smoothed absolute forecast error.

    Robust statistics (median / MAD) are used so that the anomaly itself does
    not inflate the baseline it is being compared against.

    A very well-behaved channel can have a MAD close to zero, which would make
    any deviation divide out to an absurd z-score and let one channel dominate
    every event.  The scale is therefore floored by both the channel's own
    error spread and its typical error magnitude.
    """
    diff = np.abs(y_true - y_pred)
    # Channels have different lengths once each runs at its own full extent, so
    # every statistic is computed over that channel's real readings only - a
    # NaN tail must not drag its median or spread around.
    err = np.full_like(diff, np.nan)
    for j in range(diff.shape[1]):
        valid = np.flatnonzero(~np.isnan(diff[:, j]))
        if valid.size:
            err[valid, j] = _ewma(diff[valid, j], smooth)

    # A channel can be entirely NaN - one too short for its own window, or one
    # whose model failed to load - and taking a median over nothing warns and
    # yields NaN. Such a channel simply has no score; say so deliberately rather
    # than emitting a RuntimeWarning per statistic and carrying NaN onwards.
    has_data = np.any(~np.isnan(err), axis=0)
    med = np.zeros(err.shape[1], dtype=float)
    mad = np.zeros_like(med)
    spread = np.zeros_like(med)
    if has_data.any():
        sub = err[:, has_data]
        med[has_data] = np.nanmedian(sub, axis=0)
        mad[has_data] = np.nanmedian(np.abs(sub - med[has_data]), axis=0) * 1.4826
        spread[has_data] = np.nanstd(sub, axis=0)

    floor = np.maximum(0.05 * med, 0.05 * spread)
    scale = np.maximum(np.maximum(mad, floor), 1e-6)
    z = (err - med) / scale
    z[:, ~has_data] = np.nan
    return err, z


def _longest_run(flag: np.ndarray, lo: int, hi: int) -> int:
    """Longest stretch of consecutive True values inside flag[lo:hi + 1]."""
    best = cur = 0
    for v in flag[lo:hi + 1]:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def find_sequences(flag: np.ndarray, min_run: int = MIN_RUN,
                   merge_gap: int = MERGE_GAP) -> list[tuple[int, int]]:
    """Contiguous True runs of `flag`, merged across short gaps.

    `min_run` is applied to the longest *consecutive* stretch of flagged points
    inside a merged sequence, not to the merged span.  Filtering on the span
    would let two isolated single-sample blips 40 steps apart pass a min_run of
    5 as one 41-step sequence, so the noise filter would never actually fire and
    the operator would see a 41-step anomaly built from two samples.
    """
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p <= merge_gap:
            p = i
        else:
            runs.append((s, p))
            s = p = i
    runs.append((s, p))
    return [(a, b) for a, b in runs if _longest_run(flag, a, b) >= min_run]


def dynamic_threshold(e: np.ndarray, min_run: int = MIN_RUN,
                      merge_gap: int = MERGE_GAP,
                      z_max: float | None = None) -> float:
    """Nonparametric threshold, following the telemanom paper.

    Rather than fixing a sigma level, choose the cut that most reduces the mean
    and standard deviation of the error signal per anomalous point it creates:

        argmax_eps  (dmu/mu + dsigma/sigma) / (|e_a| + |E_seq|^2)

    The penalty is linear in the number of flagged points but *quadratic* in the
    number of separate sequences, which is what stops the threshold collapsing
    into many scattered fragments while still allowing one genuinely long
    anomaly through.  (Squaring the point count instead makes the threshold so
    conservative that whole spacecraft return no detections at all.)
    """
    mu, sd = float(e.mean()), float(e.std())
    if sd < 1e-12:
        return float(e.max()) + 1.0

    best_eps, best_score = float(e.max()) + 1.0, -np.inf
    for zc in np.arange(2.0, z_max if z_max is not None else Z_SEARCH_MAX, 0.5):
        eps = mu + zc * sd
        below = e[e <= eps]
        n_above = int((e > eps).sum())
        if n_above == 0 or below.size < 2:
            continue
        d_mu = (mu - float(below.mean())) / mu if mu else 0.0
        d_sd = (sd - float(below.std())) / sd
        n_seq = max(1, len(find_sequences(e > eps, min_run, merge_gap)))
        score = (d_mu + d_sd) / (n_above + n_seq ** 2)
        if score > best_score:
            best_eps, best_score = float(eps), score
    return best_eps


def prune_sequences(seqs: list[tuple[int, int]], e: np.ndarray, eps: float,
                    min_drop: float | None = None) -> list[tuple[int, int]]:
    """Discard weak sequences, following telemanom's pruning step.

    Sequences are ranked by peak error.  Walking down that ranking, a sequence
    is kept if it - or any weaker sequence below it - is followed by a relative
    drop larger than `min_drop`; the tail after the last such drop is treated as
    normal variation.  The final comparison is against the largest sub-threshold
    error, so a lone strong sequence is judged against ordinary behaviour rather
    than discarded for having nothing to be compared with.

    Note this keeps everything above the *last* significant drop, not just up to
    the first small one - several equally strong anomalies on one channel must
    all survive, which is why recall depends on getting this right.
    """
    if min_drop is None:
        min_drop = PRUNE_DROP
    if not seqs:
        return []
    ranked = sorted(((float(e[s:t + 1].max()), (s, t)) for s, t in seqs),
                    key=lambda x: x[0], reverse=True)
    below = e[e <= eps]
    max_normal = float(below.max()) if below.size else 0.0

    keep_until = -1
    for i, (peak, _) in enumerate(ranked):
        nxt = ranked[i + 1][0] if i + 1 < len(ranked) else max_normal
        drop = (peak - nxt) / peak if peak > 0 else 0.0
        if drop > min_drop:
            keep_until = i
    return sorted(span for _, span in ranked[:keep_until + 1])


def channel_events(z: np.ndarray, channels: list[str], z_min: float | None = None,
                   min_run: int = MIN_RUN, merge_gap: int = MERGE_GAP,
                   err: np.ndarray | None = None) -> list[dict]:
    """Per-channel anomalous runs.

    Detection is done channel by channel rather than on a fleet-wide maximum.
    A fleet max lets one chronically hard-to-forecast channel raise the score
    everywhere, which both floods the operator with false alerts and makes the
    attribution meaningless - the same channel would 'explain' every event.

    With `z_min` set the cut is a fixed sigma level (used by the dashboard's
    sensitivity slider); left as None, each channel gets its own dynamic
    threshold and its sequences are pruned.
    """
    # "both" takes the union of what each signal flags. The raw error and its
    # robust z-score fail differently - the error's variance is inflated by
    # outliers elsewhere in the series, while the z-score can flatten a channel
    # whose errors are uniformly large - and which one wins turns out to be
    # mission-dependent. Taking either lets a channel be caught by whichever
    # view sees it, instead of committing every channel to one.
    if err is None:
        signals = [z]
    elif THRESHOLD_SIGNAL == "z":
        signals = [z]
    elif THRESHOLD_SIGNAL == "both":
        signals = [err, z]
    else:
        signals = [err]

    events = []
    for j, ch in enumerate(channels):
        if z_min is None:
            flagged = np.zeros(z.shape[0], dtype=bool)
            for sig in signals:
                col = sig[:, j]
                valid = np.flatnonzero(~np.isnan(col))
                if valid.size == 0:
                    continue
                col = col[: valid[-1] + 1]
                eps = dynamic_threshold(col, min_run, merge_gap)
                kept = prune_sequences(
                    find_sequences(col > eps, min_run, merge_gap), col, eps
                )
                for a, b_ in kept:
                    flagged[a:b_ + 1] = True
            seqs = find_sequences(flagged, min_run, merge_gap)
        else:
            col = z[:, j]
            valid = np.flatnonzero(~np.isnan(col))
            if valid.size == 0:
                continue
            # Use the same valid-length slice as the dynamic branch rather than
            # relying on NaN comparisons happening to be False in the padding.
            seqs = find_sequences(z[: valid[-1] + 1, j] > z_min,
                                  min_run, merge_gap)
        for s, e_ in seqs:
            k = s + int(np.argmax(z[s:e_ + 1, j]))
            events.append({
                "channel": ch, "index": j, "start": s, "end": e_, "peak": k,
                "z": float(z[k, j]),
            })
    return events


def group_incidents(events: list[dict], link_gap: int = MERGE_GAP) -> list[list[dict]]:
    """Cluster channel events that overlap in time into one operator incident.

    A real fault shows up as several channels deviating together; grouping them
    is what turns 30 raw channel alerts into a handful of incidents an operator
    can actually triage.
    """
    if not events:
        return []
    order = sorted(events, key=lambda e: e["start"])
    groups, cur, cur_end = [], [order[0]], order[0]["end"]
    for ev in order[1:]:
        if ev["start"] <= cur_end + link_gap:
            cur.append(ev)
            cur_end = max(cur_end, ev["end"])
        else:
            groups.append(cur)
            cur, cur_end = [ev], ev["end"]
    groups.append(cur)
    return groups


def attribute(group: list[dict], err: np.ndarray, y_true: np.ndarray,
              y_pred: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Explainability layer: rank the channels that make up one incident.

    Only channels that actually breached the threshold contribute, and each is
    weighted by how far past it went.  Attribution reuses the per-channel
    forecast errors the detector already computed, so it costs no extra forward
    passes - unlike SHAP or permutation importance, which is why this stays
    viable in real time.
    """
    total = sum(e["z"] for e in group) or 1.0
    onset = min(e["start"] for e in group)
    contributions = []
    for ev in sorted(group, key=lambda e: e["z"], reverse=True):
        j, k = ev["index"], ev["peak"]
        actual, pred = float(y_true[k, j]), float(y_pred[k, j])
        contributions.append({
            "channel": ev["channel"],
            "label": channel_label(ev["channel"]),
            "onset": int(ev["start"]),
            # How long after the first affected channel this one started to
            # deviate.  Ordering is what separates the channel that led the
            # fault from the ones that merely reacted to it.
            "lag": int(ev["start"] - onset),
            "subsystem": SUBSYSTEM.get(ev["channel"].split("-")[0],
                                       ev["channel"].split("-")[0]),
            "z": ev["z"],
            "share_pct": 100.0 * ev["z"] / total,
            "abs_error": float(err[ev["start"]:ev["end"] + 1, j].max()),
            "actual": actual,
            "predicted": pred,
            "deviation": actual - pred,
            "direction": "above expected" if actual > pred else "below expected",
        })

    rolled: dict[str, dict] = {}
    for c in contributions:
        r = rolled.setdefault(
            c["subsystem"],
            {"subsystem": c["subsystem"], "share_pct": 0.0, "channels": []},
        )
        r["share_pct"] += c["share_pct"]
        r["channels"].append(c["channel"])
    subsystems = sorted(rolled.values(), key=lambda r: r["share_pct"], reverse=True)

    # Order the same channels by onset rather than magnitude.  The biggest
    # deviation is often downstream of the fault; the earliest one is the
    # better candidate for its origin.
    for c in contributions:
        c["chain_rank"] = 0
    chain = sorted(contributions, key=lambda c: (c["lag"], -c["z"]))
    for rank, c in enumerate(chain):
        c["chain_rank"] = rank
    return contributions, subsystems


def detect(bundle: TelemetryBundle, z_min: float | None = None,
           min_run: int = MIN_RUN, per_channel: bool | None = None) -> dict:
    """Full detection + explanation pass over a spacecraft's test telemetry.

    `per_channel` selects the forecaster: dedicated per-channel models when
    available (better forecasts, so more anomalies clear the threshold), or the
    single joint multivariate model.  Left as None it prefers the per-channel
    ensemble and falls back to the joint model if it has not been trained.
    """
    fingerprint = _model_fingerprint(bundle.spacecraft)
    key = (bundle.spacecraft, per_channel, fingerprint)
    cached = _FORECAST_CACHE.get(key) or _load_disk(
        bundle.spacecraft, per_channel, fingerprint)
    if cached is not None:
        _FORECAST_CACHE[key] = cached
        channels, y_true, y_pred, t, forecaster, lengths = cached
    else:
        channels = list(bundle.channels)
        ensemble = (load_channel_detectors(bundle.spacecraft)
                    if per_channel is not False else None)
        if ensemble is not None and set(channels) <= set(ensemble[0]):
            nets, scalers, windows = ensemble
            y_true, y_pred, t, lengths = forecast_per_channel(
                nets, scalers, channels, bundle, windows=windows)
            forecaster = "per-channel ensemble (full length)"
        else:
            # Only load the joint model when it is actually needed: a live
            # satellite has per-channel models and no joint checkpoint at all.
            net, scaler, channels = load_detector(bundle.spacecraft)
            y_true, y_pred, t = forecast(net, scaler, bundle.test, bundle.test_cmd)
            lengths = bundle.channel_lengths()
            forecaster = "joint multivariate"
        payload = (channels, y_true, y_pred, t, forecaster, lengths)
        _FORECAST_CACHE[key] = payload
        _save_disk(bundle.spacecraft, per_channel, fingerprint, payload)
    err, z = error_scores(y_true, y_pred)

    events = channel_events(z, channels, z_min=z_min, min_run=min_run, err=err)

    anomalies = []
    for group in group_incidents(events):
        contribs, subs = attribute(group, err, y_true, y_pred)
        s = min(e["start"] for e in group)
        e_ = max(e["end"] for e in group)
        lead = max(group, key=lambda e: e["z"])
        anomalies.append(Anomaly(
            start=int(t[s]), end=int(t[e_]), peak=int(t[lead["peak"]]),
            severity=float(lead["z"]),
            contributions=contribs, subsystems=subs,
        ))
    anomalies.sort(key=lambda a: a.severity, reverse=True)

    return {
        "spacecraft": bundle.spacecraft,
        "forecaster": forecaster,
        # Readings actually available per channel - scoring must be restricted
        # to these, since anomalies beyond them were never presented.
        "channel_lengths": lengths,
        "channels": channels,
        "t": t,                 # test-series index for each scored row
        "y_true": y_true,
        "y_pred": y_pred,
        "err": err,
        "z": z,
        "fleet": np.nanmax(z, axis=1),   # display only - detection is per channel
        "events": events,         # per-channel detections, for evaluation
        "anomalies": anomalies,
    }


if __name__ == "__main__":
    # Windows consoles default to cp1252, which cannot render sigma.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from config import SPACECRAFT
    from data import load_spacecraft

    for sc in SPACECRAFT:
        try:
            result = detect(load_spacecraft(sc))
        except FileNotFoundError:
            print(f"[{sc}] no trained model yet - run train.py first")
            continue
        print(f"\n[{sc}] {len(result['anomalies'])} anomalies detected")
        for a in result["anomalies"][:5]:
            print(f"  t={a.start}-{a.end} ({a.duration:4d} steps)  {a.explanation()}")
