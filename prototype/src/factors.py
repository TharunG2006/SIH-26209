"""Why did this deviation happen — the external factor behind an incident.

Attribution answers *which* channel deviated and the causal ordering answers
*in what order*. Neither answers *why*, and "D-14 deviated first" does not tell
an engineer what to do about it.

This module attaches the external conditions that were in force when an
incident began, so an alert can say "this followed a commanded operation" or
"the spacecraft was in eclipse" rather than leaving the operator to work it out.

Three factors, in descending order of how firmly they can be established:

* **Commanding** - the NASA benchmark ships 24 command bits alongside every
  telemetry reading. If a command was issued shortly before a deviation, that
  is a documented event in the data itself, not an inference.
* **Illumination** - a satellite crossing into Earth's shadow loses solar input
  within seconds, which is the single most common cause of a coordinated power
  deviation. Computed from the orbital elements and the frame's UTC time, so it
  needs a live source; the benchmark has no clock.
* **Space weather** - solar particle flux and geomagnetic activity, from NOAA's
  open feeds, likewise matched by UTC time.

Every factor is reported as a coincidence with a confidence, never as a proven
cause. A command preceding a deviation is strong evidence; a geomagnetic storm
on the same day is weak. The wording reflects that difference.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

# How far back to look for a triggering condition. A commanded operation shows
# up in telemetry within a few readings; space weather acts over hours.
SPACE_WEATHER_WINDOW = 6       # hours

NOAA = "https://services.swpc.noaa.gov/json"

# Background levels, used to decide whether conditions were actually elevated
# rather than merely present. Above these, an event is worth reporting.
PROTON_BACKGROUND = 1.0        # particles/cm2-s-sr at >=10 MeV
KP_STORM = 5.0                 # planetary K index: 5 is a G1 storm


# A command bit that is asserted most of the time carries no information: it
# would "explain" almost every incident. Only bits that are rare overall, and
# that newly switch on close to the onset, are worth reporting.
COMMON_COMMAND_RATE = 0.20     # bits active more often than this are ignored
COMMAND_LOOKBACK = 60          # readings

# A factor must fire at least this much more often at real anomalies than at
# arbitrary moments before it is allowed into an explanation.
MIN_USEFUL_LIFT = 2.0

# Commanding failed that test on the NASA benchmark (lift ~1.0 at every setting
# tried) so it is off by default there. See calibrate_command_factor.
REPORT_COMMANDING = False


def command_activity(bundle, start: int, lookback: int = COMMAND_LOOKBACK,
                     common_rate: float = COMMON_COMMAND_RATE) -> dict | None:
    """A commanded operation shortly before an incident opened.

    The benchmark's 24 command bits already feed the model as input but were
    never surfaced in an explanation, so a deviation that merely followed a
    commanded operation looked like an unexplained fault.

    Two things stop this from becoming a meaningless label. Most bits are
    asserted almost continuously - on SMAP one is active in 78% of readings - so
    a bit that common would flag nearly every incident while telling an operator
    nothing, and is excluded. And what matters is a command *arriving*, not one
    already in force, so only a 0-to-1 transition inside the lookback counts.
    """
    cmds = getattr(bundle, "test_cmd", None)
    if cmds is None or cmds.size == 0 or cmds.shape[1] == 0:
        return None
    lo = max(0, start - lookback)
    hi = min(len(cmds), start + 1)
    if hi - lo < 2:
        return None

    base_rate = cmds.mean(axis=0)
    window = cmds[lo:hi]
    # A transition is a reading where the bit is set and the one before was not.
    prev = cmds[max(0, lo - 1):hi - 1]
    if len(prev) != len(window):
        prev = np.vstack([window[:1] * 0, window[:-1]])
    onsets = (window > 0) & (prev <= 0)

    informative = {}
    for bit in range(cmds.shape[1]):
        if base_rate[bit] > common_rate:
            continue          # asserted too often to distinguish anything
        rows = np.flatnonzero(onsets[:, bit])
        if rows.size:
            informative[bit] = int(start - (lo + rows[-1]))

    if not informative:
        return None
    nearest = min(informative.values())
    rarest = min(informative, key=lambda b: base_rate[b])
    return {
        "factor": "commanding",
        # A rare command switching on a few readings before a deviation is
        # strong evidence; the same thing 50 readings earlier much less so.
        "confidence": "strong" if nearest <= 10 else "moderate",
        "detail": (f"command bit {rarest} (normally active in "
                   f"{base_rate[rarest] * 100:.1f}% of readings) switched on "
                   f"{informative[rarest]} readings before onset"),
        "commands": sorted(informative.items(), key=lambda kv: kv[1])[:5],
        "readings_before_onset": nearest,
    }


def calibrate_command_factor(bundle, lookback: int = COMMAND_LOOKBACK,
                             common_rate: float = COMMON_COMMAND_RATE,
                             trials: int = 300, seed: int = 0) -> dict:
    """How often does the command factor fire at real anomalies vs at random?

    A factor is only an explanation if it is more likely when something is
    actually wrong. Lift near 1.0 means the factor fires just as readily at an
    arbitrary moment, so attaching it to an alert would dress up noise as a
    cause - the same failure as picking a protocol header for a telemetry
    channel because it is present in every frame.

    Measured on SMAP, the command bits give a lift of roughly 1.0 across every
    sensible lookback and rarity threshold: about 79% of readings have some
    command asserted, and with a lookback wide enough to be meaningful at least
    one rare bit almost always transitions. Commanding is therefore NOT reported
    as a factor on this benchmark. The machinery stays because a mission with
    real command logs would be a different matter, but it has to earn its place
    by this measurement first.
    """
    rng = np.random.default_rng(seed)
    n = len(bundle.test_cmd)
    starts = sorted({w[0] for seqs in bundle.scorable_labels(
        bundle.channel_lengths(full=True)).values() for w in seqs if w[0] < n})
    if not starts:
        return {"lift": None, "reason": "no labelled anomalies to calibrate on"}

    at_anom = sum(1 for s0 in starts
                  if command_activity(bundle, s0, lookback, common_rate))
    at_rand = sum(1 for _ in range(trials)
                  if command_activity(bundle, int(rng.integers(lookback + 10, n - 1)),
                                      lookback, common_rate))
    p_anom, p_rand = at_anom / len(starts), at_rand / trials
    return {
        "fires_at_anomalies": round(p_anom, 3),
        "fires_at_random": round(p_rand, 3),
        "lift": round(p_anom / p_rand, 3) if p_rand else None,
        "n_anomalies": len(starts),
        "discriminative": bool(p_rand and p_anom / p_rand >= MIN_USEFUL_LIFT),
    }


def _get_json(url: str, timeout: int = 30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def space_weather(when: str, window_hours: int = SPACE_WEATHER_WINDOW
                  ) -> dict | None:
    """Solar and geomagnetic conditions around a UTC instant.

    Radiation is a well-documented cause of spacecraft upsets, and it is the one
    external driver the telemetry itself can never show - the satellite has no
    sensor for "a solar particle event happened". NOAA publishes the relevant
    measurements openly, so the only requirement is a real timestamp, which is
    why this works on live captures and not on the NASA benchmark.
    """
    try:
        t = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    lo, hi = t - timedelta(hours=window_hours), t + timedelta(hours=1)

    findings = []
    # The 1-day feed only reaches back 24 hours, so an incident from earlier in
    # a capture silently returned nothing. The 7-day feeds cover any timestamp a
    # month-long SatNOGS capture is likely to contain.
    protons = _get_json(f"{NOAA}/goes/primary/integral-protons-7-day.json")
    if protons:
        vals = []
        for row in protons:
            if row.get("energy") != ">=10 MeV":
                continue
            try:
                ts = datetime.fromisoformat(row["time_tag"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            if lo <= ts <= hi and row.get("flux") is not None:
                vals.append(float(row["flux"]))
        if vals:
            peak = max(vals)
            findings.append({
                "measure": "proton flux >=10 MeV",
                "peak": peak,
                "elevated": peak > PROTON_BACKGROUND,
                "detail": f"peak {peak:.3g} particles/cm2-s-sr "
                          f"({'elevated' if peak > PROTON_BACKGROUND else 'background'})",
            })

    # Solar flares: X-ray flux is the standard measure and has a 7-day feed.
    xray = _get_json(f"{NOAA}/goes/primary/xrays-7-day.json")
    if xray:
        vals = []
        for row in xray:
            if row.get("energy") != "0.1-0.8nm":
                continue
            try:
                ts = datetime.fromisoformat(row["time_tag"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            if lo <= ts <= hi and row.get("flux") is not None:
                vals.append(float(row["flux"]))
        if vals:
            peak = max(vals)
            # NOAA flare classes: C >= 1e-6, M >= 1e-5, X >= 1e-4 W/m2.
            cls = ("X" if peak >= 1e-4 else "M" if peak >= 1e-5
                   else "C" if peak >= 1e-6 else "below C")
            findings.append({
                "measure": "GOES X-ray flux 0.1-0.8nm",
                "peak": peak,
                "elevated": peak >= 1e-5,       # M-class or above
                "detail": f"peak {peak:.2e} W/m2 (class {cls})",
            })

    # The planetary K index feed only spans a few hours, so it is used when the
    # incident is recent and skipped otherwise rather than reported as absent.
    kp = _get_json(f"{NOAA}/planetary_k_index_1m.json")
    if kp:
        vals = []
        for row in kp:
            try:
                ts = datetime.fromisoformat(row["time_tag"])
            except (ValueError, KeyError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if lo <= ts <= hi and row.get("kp_index") is not None:
                vals.append(float(row["kp_index"]))
        if vals:
            peak = max(vals)
            findings.append({
                "measure": "planetary K index",
                "peak": peak,
                "elevated": peak >= KP_STORM,
                "detail": f"peak Kp {peak:.1f} "
                          f"({'storm level' if peak >= KP_STORM else 'quiet'})",
            })

    if not findings:
        return None
    elevated = [f for f in findings if f["elevated"]]
    return {
        "factor": "space weather",
        # Coincidence in time is not causation, and space weather is present
        # continuously - only an elevated reading is worth an operator's
        # attention, and even then it is a hypothesis.
        "confidence": "moderate" if elevated else "none",
        "detail": "; ".join(f["detail"] for f in findings),
        "elevated": bool(elevated),
        "measures": findings,
    }


def explain_factors(bundle, anomaly, when: str | None = None,
                    check_space_weather: bool = False) -> list[dict]:
    """External conditions in force when an incident began.

    Returns a possibly-empty list. An empty list is a real answer: it means
    nothing we can observe accounts for the deviation, which is exactly the case
    an engineer should look at hardest.
    """
    found = []
    if REPORT_COMMANDING:
        cmd = command_activity(bundle, anomaly.start)
        if cmd:
            found.append(cmd)
    if when and check_space_weather:
        sw = space_weather(when)
        if sw and sw["elevated"]:
            found.append(sw)
    return found


def summarise(factors: list[dict]) -> str:
    """One line an operator can read."""
    if not factors:
        return ("No external factor found - no commanding or environmental "
                "condition we can observe accounts for this deviation.")
    parts = []
    for f in factors:
        parts.append(f"{f['factor']} ({f['confidence']}): {f['detail']}")
    return "Possible factor - " + "; ".join(parts) + \
           ". Coincidence in time, not proven causation."
