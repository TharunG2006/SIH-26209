"""The ESA Anomaly Dataset: real mission telemetry with classified anomalies.

NASA's benchmark says only *that* a window is anomalous. ESA's says *what kind*,
and separates a genuine fault from a rare but nominal event - an uncommanded
reset is an anomaly, the same reset following a telecommand is not. That
distinction is why this dataset can answer a question SMAP and MSL cannot:
whether any class of fault announces itself in advance.

Layout of a mission folder, as read by the benchmark's own preparation script:

    labels.csv         ID, Channel, StartTime, EndTime
    anomaly_types.csv  ID, Category (Anomaly / Rare Event / communication gap)
    channels/*.zip     one pickled DataFrame per parameter, DatetimeIndex

Channels are sampled irregularly and each on its own clock, so they are read one
at a time and never forced onto a shared index.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

# DATA_DIR points at the NASA benchmark folder; the ESA missions sit beside it,
# the same way satnogs.py locates its captures.
ESA_DIR = DATA_DIR.parent / "esa"


def mission_dir(mission: str = "Mission1") -> Path:
    """The extracted folder for a mission, wherever it was unpacked."""
    root = ESA_DIR / f"ESA-{mission}"
    if (root / "labels.csv").exists():
        return root
    for cand in ESA_DIR.rglob("labels.csv"):
        if mission.lower() in str(cand).lower():
            return cand.parent
    raise FileNotFoundError(
        f"no {mission} folder with labels.csv under {ESA_DIR}; "
        "download and extract it first")


def load_annotations(mission: str = "Mission1") -> pd.DataFrame:
    """Labels joined to their class, one row per annotated event."""
    d = mission_dir(mission)
    labels = pd.read_csv(d / "labels.csv")
    types = pd.read_csv(d / "anomaly_types.csv")
    merged = labels.merge(types, on="ID", how="left")
    # Annotations carry a timezone and the channel files do not, so comparing
    # them raises rather than silently misaligning. The benchmark's own
    # preparation script drops the zone (`ignoretz=True`); this matches it.
    for col in ("StartTime", "EndTime"):
        t = pd.to_datetime(merged[col], errors="coerce", utc=True)
        merged[col] = t.dt.tz_localize(None)
    return merged


def channel_names(mission: str = "Mission1") -> list[str]:
    return sorted(p.stem for p in (mission_dir(mission) / "channels").glob("*.zip"))


def load_channel(name: str, mission: str = "Mission1") -> pd.Series:
    """One parameter as a time-indexed series.

    The files are pickled DataFrames stored with a .zip extension, which is how
    the benchmark's own loader reads them; a genuine zip archive is handled too
    so a re-packed copy still works.

    Unpickling executes whatever the file says, so this trusts the archive. That
    is the format ESA publishes and there is no alternative reader, so the trust
    boundary is the Zenodo download itself: fetch it from the DOI in the README
    and do not point this at a mission folder from anywhere else.
    """
    path = mission_dir(mission) / "channels" / f"{name}.zip"
    try:
        df = pd.read_pickle(path)
    except Exception:
        with zipfile.ZipFile(path) as z:
            inner = z.namelist()[0]
            with z.open(inner) as fh:
                df = pd.read_pickle(fh)
    if isinstance(df, pd.Series):
        s = df
    else:
        col = name if name in df.columns else df.columns[0]
        s = df[col]
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    s.index = idx
    return s.sort_index()


def windows_for(name: str, ann: pd.DataFrame, series: pd.Series,
                categories=("Anomaly",)) -> list[tuple[int, int]]:
    """Annotated windows for one channel as integer index ranges.

    Times are converted to positions in that channel's own series, since every
    parameter runs on its own clock.
    """
    rows = ann[(ann["Channel"] == name) & (ann["Category"].isin(categories))]
    out = []
    for _, r in rows.iterrows():
        if pd.isna(r["StartTime"]) or pd.isna(r["EndTime"]):
            continue
        a = int(series.index.searchsorted(r["StartTime"]))
        b = int(series.index.searchsorted(r["EndTime"]))
        if b > a:
            out.append((a, min(b, len(series) - 1)))
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    mission = sys.argv[1] if len(sys.argv) > 1 else "Mission1"
    ann = load_annotations(mission)
    print(f"{mission}: {len(ann)} annotated events")
    print("\ncolumns:", list(ann.columns))
    for col in ann.columns:
        if ann[col].dtype == object and ann[col].nunique() < 25:
            print(f"\n{col}:")
            print(ann[col].value_counts().to_string())
    names = channel_names(mission)
    print(f"\n{len(names)} channels, e.g. {names[:6]}")
    print(f"channels with an Anomaly label: "
          f"{ann[ann['Category'] == 'Anomaly']['Channel'].nunique()}")
