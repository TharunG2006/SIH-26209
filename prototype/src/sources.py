"""One list of everything the system can monitor.

The dashboard and the database both need to answer "which satellites do we
have?", and until now both answered it from `config.SPACECRAFT` - a hardcoded
pair of NASA benchmark missions. Live captures were therefore invisible in the
UI and unrepresentable in the schema, even with models trained and anomalies
detected.

A source is either:

* **benchmark** - NASA SMAP/MSL. Labelled anomalies, anonymised channels, no
  wall-clock time. This is what the reported metrics are measured on.
* **live** - a SatNOGS capture. Real channel names and real UTC timestamps, but
  no ground truth, so nothing on it can be scored.

Both expose the same `TelemetryBundle`, so detection, attribution and the
causal ordering run over either without knowing which it has.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import MODEL_DIR, SPACECRAFT


@dataclass
class Source:
    key: str                 # what detect()/model files are keyed on
    label: str               # human-readable, for the UI
    kind: str                # "benchmark" or "live"
    n_channels: int
    n_readings: int
    has_labels: bool
    detail: str = ""
    norad_id: int | None = None

    @property
    def is_live(self) -> bool:
        return self.kind == "live"


def _live_meta() -> dict[str, dict]:
    """Trained live captures, keyed by satellite name.

    A live capture is distinguished from a benchmark mission by its metadata
    carrying a frame count rather than a channel-aligned matrix; the parquet is
    resolved by name so captures trained before this field existed still load.
    """
    from satnogs import SATNOGS_DIR

    out = {}
    for meta_file in MODEL_DIR.glob("*_channels.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        name = meta.get("satellite")
        if not name:
            continue          # a benchmark bundle, keyed on "spacecraft"
        path = meta.get("parquet")
        if not path:
            matches = sorted(SATNOGS_DIR.glob(f"{name}_*.parquet"))
            if not matches:
                continue
            path = str(matches[-1])
        meta["parquet"] = path
        # The capture filename carries the NORAD catalogue number, which is the
        # satellite's real-world identity and belongs in the Satellite table.
        m = re.search(r"_(\d+)\.parquet$", str(path))
        meta["norad_id"] = int(m.group(1)) if m else None
        out[name] = meta
    return out


def list_sources() -> list[Source]:
    sources = []
    for key, spec in SPACECRAFT.items():
        sources.append(Source(
            key=key, label=spec["label"], kind="benchmark",
            n_channels=len(spec["channels"]), n_readings=0, has_labels=True,
            detail="NASA benchmark - labelled anomalies, anonymised channels",
        ))
    for name, meta in _live_meta().items():
        sources.append(Source(
            key=name, label=f"{name} (live SatNOGS capture)", kind="live",
            n_channels=len(meta.get("channels", {})),
            n_readings=int(meta.get("frames", 0)), has_labels=False,
            detail="Live telemetry - real channel names and UTC timestamps, "
                   "no ground truth so nothing here is scored",
            norad_id=meta.get("norad_id"),
        ))
    return sources


def get_source(key: str) -> Source | None:
    return next((s for s in list_sources() if s.key == key), None)


def load_source(key: str):
    """Return (bundle, timestamps) for either kind of source.

    `timestamps` is None for the benchmark, which ships no clock, and an array
    of UTC strings for a live capture.
    """
    if key in SPACECRAFT:
        from data import load_spacecraft
        return load_spacecraft(key), None

    meta = _live_meta().get(key)
    if meta is None:
        raise KeyError(f"unknown source {key!r}")

    from data import TelemetryBundle
    from live import complete_frames, load_frames

    channels = list(meta["channels"])
    df = complete_frames(load_frames(meta["parquet"]), channels)
    values = df[channels].to_numpy(dtype=np.float32)
    # A live frame carries no commanding block, so the command features are
    # zero-width and the models forecast from telemetry history alone.
    cmds = np.zeros((len(df), 0), dtype=np.float32)
    bundle = TelemetryBundle(
        spacecraft=key, channels=channels,
        train=values, test=values, train_cmd=cmds, test_cmd=cmds,
        labels={c: [] for c in channels},
        full_test={c: df[c].to_numpy(dtype=np.float32) for c in channels},
        full_cmd={c: cmds for c in channels},
    )
    stamps = df["timestamp"].to_numpy() if "timestamp" in df else None
    return bundle, stamps


def utc_for(stamps, row: int) -> str | None:
    """UTC string for a row index, or None when the source has no clock."""
    if stamps is None or len(stamps) == 0:
        return None
    return str(stamps[int(np.clip(row, 0, len(stamps) - 1))])


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for s in list_sources():
        print(f"{s.kind:10s} {s.key:12s} {s.n_channels:3d} channels  {s.label}")
        print(f"           {s.detail}")
