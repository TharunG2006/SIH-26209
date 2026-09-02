"""Live telemetry ingestion from the SatNOGS ground-station network.

This is the counterpart to the NASA benchmark. Where SMAP/MSL give labelled
anomalies but anonymised channels and no clock, SatNOGS gives the opposite:
real channel names (`bcn_uhf_t` is a UHF radio temperature, `bcn_adcs_*` is
attitude control) and real UTC timestamps, but no ground truth.

Both endpoints used here are open - no API key. Observations come from the
SatNOGS Network API, the raw demodulated frames are decoded locally with the
`satnogs-decoders` package, and the numeric fields become channels the existing
detector can run on unchanged.

    python src/satnogs.py --list                  # satellites with decoders
    python src/satnogs.py --norad 68460           # fetch and decode one satellite
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from config import DATA_DIR

NETWORK = "https://network.satnogs.org/api"
DB_API = "https://db.satnogs.org/api"
SATNOGS_DIR = DATA_DIR.parent / "satnogs"

# Frames arrive only when a ground station happens to hear the spacecraft, so
# the series is irregular in wall-clock time.  We index by frame sequence and
# keep the real timestamp alongside - which is exactly the split the database
# schema already makes between `timestep` and `timestamp`.
FRAME_INDEX = "timestep"


def _get(url: str, timeout: int = 60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def available_decoders() -> set[str]:
    import satnogsdecoders
    d = os.path.join(os.path.dirname(satnogsdecoders.__file__), "decoder")
    return {f[:-3] for f in os.listdir(d)
            if f.endswith(".py") and not f.startswith("__")}


def _decoder_for(satellite_name: str, have: set[str]) -> str | None:
    """Match a SatNOGS satellite name to one of the installed decoders."""
    key = re.sub(r"[^a-z0-9]", "", satellite_name.lower())
    if key in have:
        return key
    # Names carry suffixes ("COSMO-1", "FOX-1A"); fall back to the longest
    # decoder name that is a substring, so "cosmo" matches "COSMO-1" but a
    # two-letter decoder does not match everything.
    hits = [h for h in have if h and h in key]
    return max(hits, key=len) if hits else None


def _decoder_class(name: str):
    mod = importlib.import_module(f"satnogsdecoders.decoder.{name}")
    cls_name = "".join(p.capitalize() for p in name.split("_"))
    return getattr(mod, cls_name, None)


def satellite_info(norad_id: int) -> dict | None:
    sats = _get(f"{DB_API}/satellites/?format=json&norad_cat_id={norad_id}")
    return sats[0] if sats else None


def list_decodable(limit: int = 40) -> list[dict]:
    """Recent observations whose satellite has an installed decoder."""
    have = available_decoders()
    obs = _get(f"{NETWORK}/observations/?format=json&status=good")
    seen, out = set(), []
    for o in obs[:limit]:
        nid = o.get("norad_cat_id")
        if not nid or nid in seen or not o.get("demoddata"):
            continue
        seen.add(nid)
        info = satellite_info(nid)
        if not info:
            continue
        dec = _decoder_for(info["name"], have)
        if dec:
            out.append({"norad_id": nid, "name": info["name"], "decoder": dec})
    return out


def fetch_observations(norad_id: int, pages: int = 4) -> list[dict]:
    """Recent observations for one satellite, newest first."""
    obs = []
    base = (f"{NETWORK}/observations/?format=json&status=good"
            f"&norad_cat_id={norad_id}")
    for page in range(1, pages + 1):
        # The API rejects an explicit `page=1` - it returns a single unrelated
        # record rather than the first page - so the first request omits the
        # parameter entirely and paging starts at 2.
        url = base if page == 1 else f"{base}&page={page}"
        try:
            batch = _get(url)
        except urllib.error.HTTPError:
            break            # past the last page
        if not isinstance(batch, list) or not batch:
            break
        obs.extend(batch)
    seen, unique = set(), []
    for o in obs:
        if o["id"] not in seen and o.get("demoddata"):
            seen.add(o["id"])
            unique.append(o)
    return unique


def decode_observation(obs: dict, cls) -> list[dict]:
    """Decode every demodulated frame in one observation."""
    from satnogsdecoders import decoder

    rows = []
    for d in obs.get("demoddata", []):
        try:
            raw = urllib.request.urlopen(d["payload_demod"], timeout=40).read()
            fields = decoder.get_fields(cls.from_bytes(raw))
        except Exception:
            # A frame can be truncated or corrupted in the air; one bad frame
            # must not abort the pass.
            continue
        numeric = {k: float(v) for k, v in fields.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if numeric:
            numeric["timestamp"] = obs["start"]
            numeric["observation_id"] = obs["id"]
            rows.append(numeric)
    return rows


def build_series(norad_id: int, pages: int = 4, workers: int = 8,
                 verbose: bool = True) -> pd.DataFrame:
    """Fetch, decode and assemble one satellite's telemetry into a table."""
    info = satellite_info(norad_id)
    if info is None:
        raise ValueError(f"NORAD {norad_id} not found in the SatNOGS database")
    dec = _decoder_for(info["name"], available_decoders())
    if dec is None:
        raise ValueError(f"no installed decoder matches satellite {info['name']!r}")
    cls = _decoder_class(dec)
    if cls is None:
        raise ValueError(f"decoder module {dec!r} exposes no frame class")

    obs = fetch_observations(norad_id, pages)
    if verbose:
        print(f"{info['name']} (NORAD {norad_id}): {len(obs)} observations "
              f"with frames, decoder {dec!r}")

    rows: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        for batch in pool.map(lambda o: decode_observation(o, cls), obs):
            rows.extend(batch)
    if not rows:
        raise ValueError("no frames decoded - the satellite may be transmitting "
                         "a mode this decoder does not cover")

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df[FRAME_INDEX] = np.arange(len(df))
    if verbose:
        print(f"  {len(df)} frames decoded, {df.shape[1] - 3} numeric fields")
    return df


def usable_channels(df: pd.DataFrame, min_unique: int = 3) -> list[str]:
    """Fields that actually vary.

    A field pinned to one value across every frame carries no information for a
    forecaster and would only add a channel whose error is identically zero.
    """
    skip = {"timestamp", "observation_id", FRAME_INDEX}
    return [c for c in df.columns
            if c not in skip and df[c].nunique(dropna=True) >= min_unique]


def save(df: pd.DataFrame, norad_id: int, name: str) -> "os.PathLike":
    SATNOGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    path = SATNOGS_DIR / f"{slug}_{norad_id}.parquet"
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="show recently-heard satellites that have a decoder")
    ap.add_argument("--norad", type=int, help="NORAD id to fetch and decode")
    ap.add_argument("--pages", type=int, default=4,
                    help="observation pages to pull (25 per page)")
    args = ap.parse_args()

    if args.list:
        found = list_decodable()
        print(f"{len(found)} recently-heard satellites have an installed decoder:\n")
        for f in found:
            print(f"  NORAD {f['norad_id']:>6}  {f['name']:<24} decoder={f['decoder']}")
        print("\nFetch one with:  python src/satnogs.py --norad <id>")
        raise SystemExit(0)

    if not args.norad:
        ap.error("pass --norad <id>, or --list to see what is available")

    info = satellite_info(args.norad)
    df = build_series(args.norad, pages=args.pages)
    cols = usable_channels(df)
    print(f"\n{len(cols)} channels vary and are usable as telemetry:")
    for c in cols[:20]:
        s = df[c]
        print(f"   {c:40s} range [{s.min():g}, {s.max():g}]  "
              f"{s.nunique()} distinct")
    if len(cols) > 20:
        print(f"   ... and {len(cols) - 20} more")
    path = save(df, args.norad, info["name"])
    span = f"{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
    print(f"\nsaved {len(df)} frames to {path}")
    print(f"real UTC span: {span}")
