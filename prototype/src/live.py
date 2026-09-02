"""Train and run the detector on live SatNOGS telemetry.

The NASA benchmark and a live feed need different handling in three ways, and
this module exists to keep those differences explicit rather than scattered
through the shared code:

* **Window size.** A pass yields a few thousand frames, not tens of thousands,
  so the 250-step benchmark window would leave too few training samples. Live
  models use a much shorter history.
* **Channel selection.** A decoded frame carries counters and identifiers
  alongside real telemetry. A monotonic counter is trivially forecastable and
  would inflate apparent accuracy while saying nothing about spacecraft health,
  so those are filtered out.
* **No ground truth.** Nobody publishes when a cubesat actually malfunctioned,
  so nothing here can be scored. This path demonstrates the system running on a
  real satellite; the NASA benchmark is what establishes that it works.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, REPORT_DIR
from model import ChannelScaler
from satnogs import FRAME_INDEX, SATNOGS_DIR, usable_channels

# A few thousand frames cannot support a 250-step history.
LIVE_WINDOW = 48
LIVE_STRIDE = 1
LIVE_EPOCHS = 30

# Fields that are structurally uninformative about spacecraft health.
METADATA_FIELDS = {"observation_id", FRAME_INDEX, "timestamp"}
COUNTER_HINTS = ("_ct", "count", "seq", "time_since", "sec_in", "time_stamp",
                 "sub_seconds", "packet_length", "process_id")


def complete_frames(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Keep only the frames that actually carry the selected channels.

    A satellite transmits several frame types and each carries a different
    block of fields, so a field is absent - not zero - in frames of another
    type.  On COSMO the ADCS block appears in 37% of frames, leaving the rest
    NaN for those columns.  Interpolating across that gap would invent
    telemetry, and leaving the NaNs in makes the training loss NaN, so the
    honest move is to restrict the series to frames of the type that carries
    these channels and renumber them.
    """
    keep = df.dropna(subset=channels).reset_index(drop=True)
    keep[FRAME_INDEX] = np.arange(len(keep))
    return keep


def health_channels(df: pd.DataFrame, min_unique: int = 8,
                    min_coverage: float = 0.25) -> list[str]:
    """Channels that plausibly reflect spacecraft health.

    Two things are excluded. Monotonic counters (uptime, sequence numbers,
    command tallies) are near-perfectly predictable from their own history, so
    a forecaster scores brilliantly on them while learning nothing about the
    spacecraft. Identifier-like fields are metadata, not telemetry.
    """
    out = []
    for c in usable_channels(df, min_unique=min_unique):
        if c in METADATA_FIELDS or any(h in c.lower() for h in COUNTER_HINTS):
            continue
        if df[c].notna().mean() < min_coverage:
            continue          # present in too few frames to model
        v = df[c].to_numpy(dtype=float)
        diffs = np.diff(v)
        if len(diffs) and np.all(diffs >= 0) and np.count_nonzero(diffs) > len(diffs) * 0.5:
            continue          # monotonically rising: a counter, not a reading
        out.append(c)
    return out


def load_frames(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values(FRAME_INDEX).reset_index(drop=True)


def train_live(df: pd.DataFrame, channels: list[str], name: str,
               window: int = LIVE_WINDOW, epochs: int = LIVE_EPOCHS,
               train_frac: float = 0.6, verbose: bool = True) -> dict:
    """Train one forecaster per live channel on the earliest frames.

    The split is chronological: the model learns from the oldest frames and is
    run on the newest, so it is never fitted on data it will later be judged
    against.
    """
    from train_channels import train_channel

    out_dir = MODEL_DIR / f"{name.lower()}_channels"
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = int(len(df) * train_frac)
    if cut <= window + 20:
        raise ValueError(
            f"only {cut} training frames for a window of {window}; fetch more "
            "history with `satnogs.py --days`, or lower the window")

    # Live frames carry no commanding information, so the command block is empty
    # and the model forecasts from telemetry history alone.
    cmds = np.zeros((len(df), 0), dtype=np.float32)
    meta = {"satellite": name, "window": window, "frames": len(df),
            "train_frames": cut, "channels": {}}

    for i, ch in enumerate(channels, 1):
        values = df[ch].to_numpy(dtype=np.float32)[:cut]
        state, scal, vl = train_channel(values, cmds[:cut], epochs=epochs,
                                        window=window, stride=LIVE_STRIDE)
        torch.save({"channel": ch, "model": state, "scaler": scal,
                    "window": window, "n_cmd": 0}, out_dir / f"{ch}.pt")
        meta["channels"][ch] = {"val_mse": vl}
        if verbose:
            print(f"  [{i:2d}/{len(channels)}] {ch:34s} val MSE {vl:.5f}",
                  flush=True)

    (MODEL_DIR / f"{name.lower()}_channels.json").write_text(json.dumps(meta, indent=2))
    return meta


def detect_live(df: pd.DataFrame, channels: list[str], name: str,
                min_run: int = 3) -> dict:
    """Run the trained live models over the full frame series."""
    import detect as D
    from data import TelemetryBundle

    values = df[channels].to_numpy(dtype=np.float32)
    cmds = np.zeros((len(df), 0), dtype=np.float32)
    bundle = TelemetryBundle(
        spacecraft=name, channels=list(channels),
        train=values, test=values, train_cmd=cmds, test_cmd=cmds,
        labels={c: [] for c in channels},
        full_test={c: df[c].to_numpy(dtype=np.float32) for c in channels},
        full_cmd={c: cmds for c in channels},
    )
    D.clear_forecast_cache()
    return D.detect(bundle, min_run=min_run)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?",
                    help="a file saved by satnogs.py (default: newest)")
    ap.add_argument("--window", type=int, default=LIVE_WINDOW)
    ap.add_argument("--epochs", type=int, default=LIVE_EPOCHS)
    ap.add_argument("--max-channels", type=int, default=16,
                    help="cap on channels to train, most variable first")
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()

    path = args.parquet
    if path is None:
        files = sorted(SATNOGS_DIR.glob("*.parquet"),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            raise SystemExit("no SatNOGS captures found - run satnogs.py first")
        path = files[-1]
    name = re.sub(r"_\d+$", "", str(path).split("\\")[-1].split("/")[-1][:-8])

    df = load_frames(path)
    chans = health_channels(df)
    # Prefer the most variable channels: a near-flat signal contributes little
    # and each extra channel is another model to train.
    chans = sorted(chans, key=lambda c: df[c].std(), reverse=True)[:args.max_channels]
    raw_frames = len(df)
    df = complete_frames(df, chans)
    print(f"{name}: {raw_frames} frames decoded, {len(df)} carry the selected "
          f"channels, {len(chans)} health channels")
    for c in chans:
        print(f"   {c:38s} std={df[c].std():.4g}")

    if not args.detect_only:
        print("\ntraining live forecasters")
        train_live(df, chans, name, window=args.window, epochs=args.epochs)

    print("\ndetecting")
    result = detect_live(df, chans, name)
    ts = df["timestamp"].to_numpy()
    print(f"{len(result['anomalies'])} incidents on live telemetry\n")
    for a in result["anomalies"][:8]:
        when = ts[min(a.start, len(ts) - 1)]
        print(f"  frame {a.start}-{a.end}  {when}")
        print(f"     {a.explanation()}")
        print(f"     {a.propagation()}")
    (REPORT_DIR / f"live_{name.lower()}.json").write_text(json.dumps({
        "satellite": name, "frames": len(df), "channels": chans,
        "incidents": [{"start": a.start, "end": a.end,
                       "severity": a.severity,
                       "timestamp": str(ts[min(a.start, len(ts) - 1)]),
                       "explanation": a.explanation(),
                       "propagation": a.propagation()}
                      for a in result["anomalies"]],
    }, indent=2))
    print(f"\nwrote {REPORT_DIR / f'live_{name.lower()}.json'}")
