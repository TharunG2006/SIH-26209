"""Train one dedicated forecaster per telemetry channel.

The joint multivariate model gives every channel a comparable error scale,
which is what makes attribution possible - but one shared representation has to
cover 27 very different signals, and it forecasts most of them poorly.  An
anomaly cannot stand out against predictions that are chronically wrong, so
recall is capped by forecast quality rather than by thresholding.

This module trains a small model per channel (the arrangement NASA's telemanom
uses) so each channel gets a forecast good enough for its anomalies to show.
Detection then runs on these; attribution still works because the z-scores are
normalised per channel and so remain comparable.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import LR, MODEL_DIR, PATIENCE, SEED, SPACECRAFT, WINDOW
from data import load_spacecraft
from model import ChannelScaler, TelemetryForecaster

# Per-channel models are small: one signal to learn, so capacity that helps the
# joint model here just overfits a few thousand timesteps.
CH_HIDDEN = 80
CH_LAYERS = 2
CH_DROPOUT = 0.3
CH_EPOCHS = 20
CH_BATCH = 128

# Consecutive sliding windows share 249 of their 250 timesteps, so training on
# every one is almost entirely redundant computation.  Taking every 4th window
# cuts training time roughly fourfold with no meaningful loss of coverage - the
# discarded windows are near-duplicates of the ones kept.  Evaluation still
# scores every timestep; this stride only thins the training set.
CH_STRIDE = 4

# A CPU LSTM barely benefits from extra intra-op threads at this batch size, so
# the machine is used far better by training several channels side by side with
# a couple of threads each than by training one channel with all of them.
# Fewer workers than cores. Each holds a full-length series plus its model, and
# with eight of them a worker was killed outright mid-run (BrokenProcessPool)
# after 54 of 82 channels - the pool gives no warning before that happens.
WORKERS = max(1, min(5, (os.cpu_count() or 4) // 3))
THREADS_PER_WORKER = 2


def _windows(values: np.ndarray, cmds: np.ndarray, window: int,
             stride: int = 1):
    """Sliding windows for a single channel: history + commands -> next value."""
    feats = np.concatenate([values.reshape(-1, 1), cmds], axis=1)
    idx = np.arange(0, len(values) - window, stride)
    X = np.stack([feats[i:i + window] for i in idx])
    y = values[idx + window].reshape(-1, 1)
    return X.astype(np.float32), y.astype(np.float32)


def train_channel(values: np.ndarray, cmds: np.ndarray, epochs: int = CH_EPOCHS,
                  val_frac: float = 0.15, window: int | None = None,
                  stride: int | None = None) -> tuple[dict, dict, float]:
    """Train one channel's forecaster.

    `window` is explicit so live feeds can use a shorter history than the
    250-step benchmark setting: a SatNOGS pass yields a few thousand frames, and
    a 250-step window would leave too few training samples to learn anything.
    """
    torch.manual_seed(SEED)
    scaler = ChannelScaler().fit(values.reshape(-1, 1))
    scaled = scaler.transform(values.reshape(-1, 1)).ravel()

    window = WINDOW if window is None else window
    stride = CH_STRIDE if stride is None else stride
    X, y = _windows(scaled, cmds, window, stride=stride)
    cut = max(1, int(len(X) * (1 - val_frac)))
    tr = TensorDataset(torch.from_numpy(X[:cut]), torch.from_numpy(y[:cut]))
    va = TensorDataset(torch.from_numpy(X[cut:]), torch.from_numpy(y[cut:]))
    tr_dl = DataLoader(tr, batch_size=CH_BATCH, shuffle=True)
    va_dl = DataLoader(va, batch_size=CH_BATCH)

    # The command block is 24 wide on the NASA benchmark and empty on a live
    # feed, so the input width is taken from the data rather than assumed.
    net = TelemetryForecaster(1, n_cmd=cmds.shape[1], hidden=CH_HIDDEN,
                              layers=CH_LAYERS, dropout=CH_DROPOUT)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = torch.nn.MSELoss()

    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        net.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                vl += lossf(net(xb), yb).item() * len(xb)
        vl /= max(1, len(va))

        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is None:
        # Every epoch's validation loss was NaN or infinite - almost always NaN
        # in the input rather than a training failure.
        raise ValueError(
            "training never produced a valid model; check the channel for NaN "
            f"(input had {int(np.isnan(values).sum())} NaN of {values.size})")
    net.load_state_dict(best_state)
    return net.state_dict(), scaler.state_dict(), float(best)


def _train_one(job):
    """Worker entry point - must be importable at module level for Windows."""
    channel, values, cmds, epochs, window = job
    torch.set_num_threads(THREADS_PER_WORKER)
    state, scal, vl = train_channel(values, cmds, epochs=epochs, window=window)
    return channel, state, scal, vl, window


def train_spacecraft_channels(name: str, epochs: int = CH_EPOCHS,
                              verbose: bool = True, force: bool = False) -> dict:
    bundle = load_spacecraft(name)
    out_dir = MODEL_DIR / f"{name.lower()}_channels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train on each channel's own full series rather than the aligned matrix,
    # which only covers the subset that shares a timeline. A channel with a
    # short excerpt gets a proportionally shorter window so it still has enough
    # training samples to learn from.
    # Skip channels already on disk. A long run is worth resuming rather than
    # restarting: a crash at channel 54 should cost one model, not fifty-four.
    jobs, skipped = [], 0
    for ch in bundle.channels:
        if not force and (out_dir / f"{ch}.pt").exists():
            skipped += 1
            continue
        values = bundle.full_train[ch]
        cmds = bundle.full_train_cmd[ch]
        window = min(WINDOW, max(24, len(values) // 4))
        jobs.append((ch, values, cmds, epochs, window))
    if verbose and skipped:
        print(f"  {skipped} channels already trained, {len(jobs)} to go",
              flush=True)

    meta = {"spacecraft": name, "window": WINDOW, "hidden": CH_HIDDEN,
            "stride": CH_STRIDE, "workers": WORKERS, "channels": {}}
    t0 = time.time()
    done = 0
    def run(batch, workers):
        """Train a batch, returning whatever completed before any failure."""
        nonlocal done
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            for ch, state, scal, vl, window in pool.map(_train_one, batch):
                torch.save({"channel": ch, "model": state, "scaler": scal,
                            "window": window, "n_cmd": 24},
                           out_dir / f"{ch}.pt")
                meta["channels"][ch] = {"val_mse": vl, "window": window}
                done += 1
                if verbose:
                    print(f"  [{done:2d}/{len(jobs)}] {ch:6s} val MSE {vl:.5f}",
                          flush=True)

    # Submit in chunks so a pool that dies takes one chunk with it, not the
    # whole run, and fall back to a single worker for anything that failed.
    remaining = list(jobs)
    for i in range(0, len(jobs), WORKERS * 2):
        chunk = jobs[i:i + WORKERS * 2]
        try:
            run(chunk, WORKERS)
            remaining = [j for j in remaining if j not in chunk]
        except cf.process.BrokenProcessPool:
            if verbose:
                print("  worker pool died - retrying this chunk serially",
                      flush=True)
            for job in chunk:
                if (out_dir / f"{job[0]}.pt").exists():
                    continue
                try:
                    run([job], 1)
                except Exception as exc:
                    print(f"  SKIP {job[0]}: {type(exc).__name__}", flush=True)

    meta["seconds"] = round(time.time() - t0, 1)
    (MODEL_DIR / f"{name.lower()}_channels.json").write_text(
        json.dumps(meta, indent=2))
    if verbose:
        print(f"  {len(jobs)} channel models in {meta['seconds']}s "
              f"({WORKERS} workers)")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", default="all", choices=["all", *SPACECRAFT])
    ap.add_argument("--epochs", type=int, default=CH_EPOCHS)
    ap.add_argument("--force", action="store_true",
                    help="retrain channels that already have a checkpoint")
    args = ap.parse_args()
    for sc in (list(SPACECRAFT) if args.spacecraft == "all" else [args.spacecraft]):
        print(f"[{sc}] training per-channel forecasters")
        train_spacecraft_channels(sc, epochs=args.epochs, force=args.force)
