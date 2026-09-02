"""Central configuration for the explainable satellite anomaly detector."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "smap_msl"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
for _d in (MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Spacecraft definitions.
#
# NASA anonymises the telemanom channel names, so we keep the original channel
# ids and expose the leading letter as a "subsystem group".  Channels listed
# here are the ones whose train/test excerpts cover a common timeline, which is
# what lets us model them jointly as one multivariate stream instead of 82
# independent univariate ones.
# ---------------------------------------------------------------------------
SPACECRAFT = {
    "SMAP": {
        "label": "SMAP (Soil Moisture Active Passive)",
        # Every channel NASA labelled for this mission. The
        # per-channel ensemble needs no shared timeline, so there is
        # no reason to monitor only the subset that happens to align.
        "channels": [
            "A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7", "A-8",
            "A-9", "B-1", "D-1", "D-11", "D-12", "D-13", "D-2",
            "D-3", "D-4", "D-5", "D-6", "D-7", "D-8", "D-9", "E-1",
            "E-10", "E-11", "E-12", "E-13", "E-2", "E-3", "E-4",
            "E-5", "E-6", "E-7", "E-8", "E-9", "F-1", "F-2", "F-3",
            "G-1", "G-2", "G-3", "G-4", "G-6", "G-7", "P-1", "P-2",
            "P-3", "P-4", "P-7", "R-1", "S-1", "T-1", "T-2",
            "T-3"
        ],
        # The subset covering a common window, which is all the joint
        # multivariate model can use - it needs one aligned matrix.
        "aligned": [
            "A-1", "A-7", "D-1", "D-2", "D-3", "D-4", "E-1", "E-2",
            "E-3", "E-4", "E-6", "E-8", "E-9", "E-10", "E-11",
            "E-12", "E-13", "F-1", "F-2", "F-3", "G-1", "G-6",
            "P-1", "P-3", "T-1", "T-2", "T-3"
        ],
    },
    "MSL": {
        "label": "MSL (Mars Science Laboratory / Curiosity)",
        # Every channel NASA labelled for this mission. The
        # per-channel ensemble needs no shared timeline, so there is
        # no reason to monitor only the subset that happens to align.
        "channels": [
            "C-1", "C-2", "D-14", "D-15", "D-16", "F-4", "F-5",
            "F-7", "F-8", "M-1", "M-2", "M-3", "M-4", "M-5", "M-6",
            "M-7", "P-10", "P-11", "P-14", "P-15", "S-2", "T-12",
            "T-13", "T-4", "T-5", "T-8", "T-9"
        ],
        # The subset covering a common window, which is all the joint
        # multivariate model can use - it needs one aligned matrix.
        "aligned": [
            "C-1", "D-14", "D-15", "D-16", "F-4", "F-5", "F-7",
            "F-8", "M-1", "M-2", "M-3", "M-4", "M-5", "M-6", "M-7",
            "P-10", "P-11", "P-14", "P-15", "T-4", "T-5"
        ],
    },
}

def _dedupe(seq):
    """Preserve order, drop repeats.

    NASA's labeled_anomalies.csv lists P-2 twice. A duplicated channel is not
    cosmetic: it would be forecast twice and emit each of its detections twice,
    inflating both the detection count and the false-alarm count, and it makes
    the number of trained models permanently one short of the number expected -
    which is enough to stall a pipeline that waits for training to finish.
    """
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


for _spec in SPACECRAFT.values():
    _spec["channels"] = _dedupe(_spec["channels"])
    if "aligned" in _spec:
        _spec["aligned"] = _dedupe(_spec["aligned"])


# Default subsystem group names.  These are deliberately neutral: the leading
# letter of a telemanom channel id is an anonymised group tag, NOT a documented
# subsystem, so calling "P" a power bus would be inventing domain knowledge the
# dataset does not contain.
SUBSYSTEM = {k: f"Group {k}" for k in "ABCDEFGMPRST"}

# ---------------------------------------------------------------------------
# Operator-supplied names.
#
# A real mission knows what its channels are; NASA's public benchmark does not
# publish that mapping.  Rather than guess, we read an optional
# config/channel_names.json (see channel_names.example.json) and fall back to
# the raw channel ids for anything unmapped.
# ---------------------------------------------------------------------------
NAMES_FILE = ROOT / "config" / "channel_names.json"


def _load_names() -> tuple[dict[str, str], dict[str, str]]:
    if not NAMES_FILE.exists():
        return {}, {}
    import json
    blob = json.loads(NAMES_FILE.read_text(encoding="utf-8"))
    return blob.get("channels", {}) or {}, blob.get("subsystems", {}) or {}


CHANNEL_NAMES, SUBSYSTEM_NAMES = _load_names()
SUBSYSTEM = {**SUBSYSTEM, **SUBSYSTEM_NAMES}


def channel_label(channel: str) -> str:
    """Display name for a channel: operator's name if supplied, else the id."""
    alias = CHANNEL_NAMES.get(channel)
    return f"{channel} — {alias}" if alias else channel


def has_operator_names() -> bool:
    """True when a real channel-name mapping has been supplied."""
    return bool(CHANNEL_NAMES or SUBSYSTEM_NAMES)

# ---------------------------------------------------------------------------
# Model / detection hyperparameters
# ---------------------------------------------------------------------------
WINDOW = 250          # timesteps of history fed to the LSTM
                      # (matches telemanom; shorter windows left several
                      #  channels badly forecast, which hid their anomalies)
HORIZON = 1           # predict the next timestep for every channel
HIDDEN = 256
LAYERS = 2
DROPOUT = 0.2
BATCH = 64
EPOCHS = 60
LR = 1e-3
PATIENCE = 6
SEED = 42

# Detection: telemanom-style smoothed errors + dynamic threshold
SMOOTH_WINDOW = 30    # EWMA span applied to per-channel errors
Z_MIN = 2.5           # minimum z-score for a point to count as anomalous
MIN_RUN = 5           # consecutive anomalous points required to open a sequence
MERGE_GAP = 50        # merge sequences separated by fewer timesteps than this
PRUNE_DROP = 0.13     # telemanom's pruning ratio: min relative drop between
                      # consecutive sequence peaks before the rest are dropped
