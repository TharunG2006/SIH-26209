"""Train the multivariate LSTM forecaster for one spacecraft."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import (BATCH, EPOCHS, LR, MODEL_DIR, PATIENCE, SEED, SPACECRAFT,
                    WINDOW)
from data import load_spacecraft, make_windows
from model import ChannelScaler, TelemetryForecaster


def train_spacecraft(name: str, epochs: int = EPOCHS, val_frac: float = 0.15,
                     verbose: bool = True) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    bundle = load_spacecraft(name)
    scaler = ChannelScaler().fit(bundle.train)

    X, y, _ = make_windows(scaler.transform(bundle.train), bundle.train_cmd, WINDOW)

    # Chronological split: the tail of the training excerpt is held out, so we
    # never validate on data that precedes what the model was fitted on.
    cut = int(len(X) * (1 - val_frac))
    tr = TensorDataset(torch.from_numpy(X[:cut]), torch.from_numpy(y[:cut]))
    va = TensorDataset(torch.from_numpy(X[cut:]), torch.from_numpy(y[cut:]))
    tr_dl = DataLoader(tr, batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(va, batch_size=BATCH)

    net = TelemetryForecaster(bundle.n_channels)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = torch.nn.MSELoss()

    best, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        tl = 0.0
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(tr)

        net.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                vl += lossf(net(xb), yb).item() * len(xb)
        vl /= len(va)
        sched.step(vl)
        history.append({"epoch": ep, "train_mse": tl, "val_mse": vl})

        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"  epoch {ep:3d}  train {tl:.5f}  val {vl:.5f}"
                  f"{'  *' if bad == 0 else ''}")
        if bad >= PATIENCE:
            if verbose:
                print(f"  early stop at epoch {ep} (best val {best:.5f})")
            break

    net.load_state_dict(best_state)
    ckpt = MODEL_DIR / f"{name.lower()}_lstm.pt"
    torch.save(
        {
            "spacecraft": name,
            "channels": bundle.channels,
            "window": WINDOW,
            "model": net.state_dict(),
            "scaler": scaler.state_dict(),
        },
        ckpt,
    )
    meta = {
        "spacecraft": name,
        "channels": bundle.n_channels,
        "train_windows": int(cut),
        "val_windows": int(len(X) - cut),
        "best_val_mse": float(best),
        "epochs_run": len(history),
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }
    (MODEL_DIR / f"{name.lower()}_train.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"  saved {ckpt.name}  best val MSE {best:.5f}  ({meta['seconds']}s)")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", default="all", choices=["all", *SPACECRAFT])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    targets = list(SPACECRAFT) if args.spacecraft == "all" else [args.spacecraft]
    for sc in targets:
        print(f"[{sc}] training multivariate LSTM forecaster")
        train_spacecraft(sc, epochs=args.epochs)
