"""What each telemetry channel actually measures.

An alert that names `uhf_rf_chip_act_temperature` is precise but not readable.
An operator wants "UHF radio temperature", and wants to know a reading is in
degrees rather than millivolts.

The satnogs-decoders package carries no units metadata - the compiled decoders
expose field names and nothing else, no `unit` or `doc` attribute - so the
information has to be recovered. Two steps, in order of how much they can be
trusted:

1. **Quantity from the name.** These decoders follow the satellite's published
   telemetry format, so a field ending `_t` or containing `temperature` is a
   temperature, `volt` is a voltage, `amp` a current. This is an inference from
   a naming convention, not published metadata, and is labelled as such.

2. **Unit from the observed magnitude.** A field named `volt` could be volts or
   millivolts, and the name cannot say which. A battery bus reading 8.1 is volts;
   the same bus reading 8100 is millivolts. The scale is therefore resolved
   against the values actually present rather than assumed, which is the part a
   name alone cannot give.

Nothing here is invented: a channel whose quantity cannot be inferred is
reported as unknown rather than guessed at.
"""
from __future__ import annotations

import re

import numpy as np

# Ordered: the first pattern that matches wins, so the more specific ones come
# first. `_t` is last because it is the weakest signal.
QUANTITY_PATTERNS: list[tuple[str, str]] = [
    (r"temperature|_temp\b|_temp_|thermistor", "temperature"),
    (r"rssi|signal_strength|_snr\b", "signal strength"),
    (r"power\b|_pwr\b|watt", "power"),
    (r"volt|_v\b|voltage", "voltage"),
    (r"amp\b|_amp_|current|_ma\b|_i\b", "current"),
    (r"pos_ecef|position|_lat\b|_lon\b|altitude", "position"),
    (r"bod_rt|rot_rate|gyro|angular", "rotation rate"),
    (r"att_resid", "pointing error"),
    (r"quaternion|_att_|attitude", "attitude"),
    (r"pressure|_bar\b", "pressure"),
    (r"_t$|_t_", "temperature"),
]

# Plausible ranges for a quantity in its base unit. A reading far outside these
# says the channel is reporting a scaled unit - milli- or centi- - which the
# name never states.
SCALES: dict[str, list[tuple[float, str, float]]] = {
    # (upper bound of |median|, unit label, multiplier to reach the base unit)
    "voltage": [(60, "V", 1.0), (60_000, "mV", 1e-3), (float("inf"), "µV", 1e-6)],
    "current": [(20, "A", 1.0), (20_000, "mA", 1e-3), (float("inf"), "µA", 1e-6)],
    "power": [(500, "W", 1.0), (500_000, "mW", 1e-3), (float("inf"), "µW", 1e-6)],
    "temperature": [(200, "°C", 1.0), (float("inf"), "m°C", 1e-3)],
    "signal strength": [(float("inf"), "dBm", 1.0)],
    "pressure": [(float("inf"), "Pa", 1.0)],
}

# A unit stated outright in the field name. This beats magnitude inference:
# `psu_bat_temp_kelvin` says kelvin, and no amount of looking at the numbers
# should override the satellite telling us directly.
#
# Abbreviations are anchored to a token boundary. A bare substring would
# relabel unrelated channels with a confident-looking unit: `_ma` matches
# `_max` and `_magnetometer`, and `_k` matches almost anything.
EXPLICIT_UNITS: list[tuple[str, str]] = [
    (r"kelvin|(?:^|_)k(?:$|_)", "K"),
    (r"celsius|degc", "°C"),
    (r"(?:^|_)mv(?:$|_)|millivolt", "mV"),
    (r"(?:^|_)ma(?:$|_)|milliamp", "mA"),
    (r"(?:^|_)mw(?:$|_)|milliwatt", "mW"),
    (r"dbm", "dBm"),
]

# Frame-type prefixes that are not subsystems. `bcn_pld_sb_1_t` is a payload
# sensor carried in a beacon frame, so taking the first recognised token would
# label every beacon field "Beacon" and hide which subsystem it came from.
FRAME_PREFIXES = {"bcn", "beacon", "tlm", "hk"}

# Subsystem from the field prefix these decoders use.
SUBSYSTEM_PREFIXES = {
    "psu": "Power supply",
    "eps": "Power supply",
    "uhf": "UHF radio",
    "vhf": "VHF radio",
    "adcs": "Attitude control",
    "bcn": "Beacon",
    "obc": "Onboard computer",
    "pld": "Payload",
    "thermal": "Thermal",
    "gps": "Navigation",
}


def quantity_of(channel: str) -> str | None:
    """What this channel measures, inferred from its name."""
    name = channel.lower()
    for pattern, quantity in QUANTITY_PATTERNS:
        if re.search(pattern, name):
            return quantity
    return None


def subsystem_of(channel: str) -> str | None:
    """Which subsystem the channel belongs to, from its prefix.

    These names run from general to specific, so where two subsystems appear the
    later one is the actual source: `bcn_adcs_gps_pos_ecef_1` is a reading from
    the GPS receiver, which attitude control happens to consume. Taking the
    first match labelled it "Attitude control position", which reads as though
    the spacecraft were reporting its orientation rather than its location.
    """
    parts = [p for p in channel.lower().split("_") if p not in FRAME_PREFIXES]
    found = [SUBSYSTEM_PREFIXES[p] for p in parts[:3] if p in SUBSYSTEM_PREFIXES]
    return found[-1] if found else None


def unit_of(channel: str, values=None) -> tuple[str | None, str]:
    """Unit for a channel, resolved against its observed magnitude.

    Returns (unit, basis) where basis records how firmly it is known:
    "magnitude" when the observed values chose between candidate scales,
    "name" when the quantity was recognised but no values were supplied, and
    "unknown" when the quantity itself could not be inferred.
    """
    # A unit written into the name is authoritative; only fall back to
    # inferring one from the magnitudes when the name does not say.
    name = channel.lower()
    for pattern, unit in EXPLICIT_UNITS:
        if re.search(pattern, name):
            return unit, "name (explicit)"

    quantity = quantity_of(channel)
    if quantity is None:
        return None, "unknown"
    candidates = SCALES.get(quantity)
    if not candidates:
        return None, "unknown"
    if values is None:
        return candidates[0][1], "name"

    try:
        v = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        # A non-numeric column (a timestamp, a callsign) has no magnitude to
        # resolve a unit against; the name is all there is.
        return candidates[0][1], "name"
    v = v[np.isfinite(v)]
    if v.size == 0:
        return candidates[0][1], "name"
    magnitude = float(np.median(np.abs(v)))
    for upper, unit, _ in candidates:
        if magnitude < upper:
            return unit, "magnitude"
    return candidates[-1][1], "magnitude"


def describe(channel: str, values=None) -> dict:
    """Everything recoverable about one channel, with its provenance."""
    quantity = quantity_of(channel)
    subsystem = subsystem_of(channel)
    unit, basis = unit_of(channel, values)
    if quantity and subsystem:
        label = f"{subsystem} {quantity}"
    elif quantity:
        label = quantity.capitalize()
    else:
        label = channel
    return {
        "channel": channel,
        "quantity": quantity,
        "unit": unit,
        "unit_basis": basis,
        "subsystem": subsystem,
        "label": label,
        # Everything here is inferred from a naming convention and the observed
        # values, not read from published metadata. The dashboard says so.
        "inferred": True,
    }


def describe_all(channels, frame=None) -> dict[str, dict]:
    """Describe every channel, using the data for unit resolution when given."""
    out = {}
    for ch in channels:
        vals = None
        if frame is not None and ch in getattr(frame, "columns", []):
            vals = frame[ch].to_numpy()
        out[ch] = describe(ch, vals)
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import pandas as pd

    from satnogs import SATNOGS_DIR

    for path in sorted(SATNOGS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        print(f"\n=== {path.stem} ===")
        described = describe_all(
            [c for c in df.columns if c not in ("timestamp", "timestep",
                                                "observation_id")], df)
        known = [d for d in described.values() if d["quantity"]]
        print(f"{len(known)} of {len(described)} channels classified\n")
        for d in known[:14]:
            print(f"  {d['channel']:40s} -> {d['label']:28s} "
                  f"{str(d['unit']):5s} ({d['unit_basis']})")
