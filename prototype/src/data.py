"""Loading, alignment and windowing of the NASA SMAP/MSL telemetry."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import DATA_DIR, SPACECRAFT, SUBSYSTEM, WINDOW


@dataclass
class TelemetryBundle:
    """One spacecraft's aligned multivariate telemetry."""

    spacecraft: str
    channels: list[str]
    train: np.ndarray          # (T_train, C) telemetry values
    test: np.ndarray           # (T_test,  C) telemetry values
    train_cmd: np.ndarray      # (T_train, 24) command one-hots (union over channels)
    test_cmd: np.ndarray       # (T_test,  24)
    labels: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # Untruncated per-channel test series.  The aligned `test` matrix above is
    # required by the joint multivariate model, but the per-channel ensemble has
    # no such constraint, so it can use every reading a channel actually has.
    full_test: dict[str, np.ndarray] = field(default_factory=dict)
    full_cmd: dict[str, np.ndarray] = field(default_factory=dict)
    full_train: dict[str, np.ndarray] = field(default_factory=dict)
    full_train_cmd: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def subsystem_of(self, channel: str) -> str:
        return SUBSYSTEM.get(channel.split("-")[0], channel.split("-")[0])

    def channel_lengths(self, full: bool = False) -> dict[str, int]:
        """How many test readings are available per channel."""
        if full and self.full_test:
            return {ch: len(self.full_test[ch]) for ch in self.channels}
        n = self.test.shape[0]
        return {ch: n for ch in self.channels}

    def scorable_labels(self, lengths: dict[str, int] | None = None
                        ) -> dict[str, list[tuple[int, int]]]:
        """Labelled windows lying inside the data the detector actually saw.

        A detector cannot be charged with missing anomalies in readings it was
        never given, so scoring is restricted to what was in range.  Which
        windows qualify depends on the forecaster: the joint model sees the
        truncated matrix, the per-channel ensemble sees each channel in full.
        """
        if lengths is None:
            lengths = self.channel_lengths()
        # Forecasting only begins at WINDOW, so a window lying entirely in the
        # lead-in has no prediction to deviate from and can never be detected;
        # counting it as a miss would understate recall for a reason unrelated
        # to the detector.
        return {ch: [w for w in seqs
                     if w[1] >= WINDOW and w[0] < lengths.get(ch, 0)]
                for ch, seqs in self.labels.items()}

    def dropped_label_count(self, lengths: dict[str, int] | None = None) -> int:
        """How many labelled windows fall outside the data seen."""
        if lengths is None:
            lengths = self.channel_lengths()
        return sum(1 for ch, seqs in self.labels.items() for w in seqs
                   if w[1] < WINDOW or w[0] >= lengths.get(ch, 0))

    def label_mask(self) -> np.ndarray:
        """(T_test, C) boolean mask of the ground-truth anomaly windows."""
        mask = np.zeros(self.test.shape, dtype=bool)
        for j, ch in enumerate(self.channels):
            for s, e in self.labels.get(ch, []):
                s, e = max(0, s), min(self.test.shape[0] - 1, e)
                if s <= e:
                    mask[s : e + 1, j] = True
        return mask


def _read_channel(split: str, channel: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / split / f"{channel}.parquet")


def load_labels() -> dict[str, list[tuple[int, int]]]:
    """Ground-truth anomaly windows from NASA's labeled_anomalies.csv."""
    df = pd.read_csv(DATA_DIR / "labeled_anomalies.csv")
    out: dict[str, list[tuple[int, int]]] = {}
    for _, row in df.iterrows():
        seqs = ast.literal_eval(row["anomaly_sequences"])
        out[row["chan_id"]] = [(int(a), int(b)) for a, b in seqs]
    return out


def load_spacecraft(name: str) -> TelemetryBundle:
    """Stack every channel of one spacecraft into an aligned (T, C) matrix.

    The telemanom excerpts for these channels cover the same acquisition
    window, so we truncate each split to the shortest channel to get a common
    timeline.

    Truncation is at the tail, and it is NOT lossless: on MSL the channels run
    to 6100 readings but the shortest stops at 2038, so 7 of the 26 labelled
    anomaly windows fall past the retained range.  Those windows are excluded
    from scoring by `scorable_labels()` - a detector cannot be charged with
    missing anomalies in data it was never given.  SMAP loses nothing this way.
    """
    channels = SPACECRAFT[name]["channels"]
    # Only the joint multivariate model needs one aligned matrix, and aligning
    # all 55 SMAP channels would truncate every one of them to the shortest
    # (312 readings) and throw away almost everything. The matrix is therefore
    # built from the aligning subset, while the per-channel ensemble - which
    # needs no shared timeline - sees every channel at its full length.
    aligned = SPACECRAFT[name].get("aligned", channels)
    tr_frames = {c: _read_channel("train", c) for c in channels}
    te_frames = {c: _read_channel("test", c) for c in channels}

    n_tr = min(len(tr_frames[c]) for c in aligned)
    n_te = min(len(te_frames[c]) for c in aligned)

    train = np.column_stack([tr_frames[c]["value"].to_numpy()[:n_tr] for c in aligned])
    test = np.column_stack([te_frames[c]["value"].to_numpy()[:n_te] for c in aligned])

    cmd_cols = [f"cmd_{i}" for i in range(24)]
    # A command issued on any channel is a spacecraft-level event, so take the
    # element-wise max across channels rather than duplicating 24 columns each.
    train_cmd = np.max(
        np.stack([tr_frames[c][cmd_cols].to_numpy()[:n_tr] for c in aligned]), axis=0
    )
    test_cmd = np.max(
        np.stack([te_frames[c][cmd_cols].to_numpy()[:n_te] for c in aligned]), axis=0
    )

    all_labels = load_labels()
    labels = {c: all_labels.get(c, []) for c in channels}

    # A command is a spacecraft-level event, so the feature the models were
    # trained on is the element-wise max across channels.  Build that same union
    # across the *full* length: at each timestep it covers every channel that
    # still has data there.  Splicing a fleet union onto a single channel's own
    # columns partway through would hand the network two different feature
    # distributions in one series, and the seam would land exactly in the tail
    # that the full-length path was added to recover.
    full_len = max(len(f) for f in te_frames.values())
    cmd_union = np.zeros((full_len, len(cmd_cols)), dtype=np.float32)
    for f in te_frames.values():
        own = f[cmd_cols].to_numpy().astype(np.float32)
        cmd_union[: len(own)] = np.maximum(cmd_union[: len(own)], own)

    full_test, full_cmd = {}, {}
    for c in channels:
        f = te_frames[c]
        full_test[c] = f["value"].to_numpy().astype(np.float32)
        full_cmd[c] = cmd_union[: len(f)].copy()

    full_train, full_train_cmd = {}, {}
    tr_len = max(len(f) for f in tr_frames.values())
    tr_union = np.zeros((tr_len, len(cmd_cols)), dtype=np.float32)
    for f in tr_frames.values():
        own = f[cmd_cols].to_numpy().astype(np.float32)
        tr_union[: len(own)] = np.maximum(tr_union[: len(own)], own)
    for c in channels:
        full_train[c] = tr_frames[c]["value"].to_numpy().astype(np.float32)
        full_train_cmd[c] = tr_union[: len(tr_frames[c])].copy()

    return TelemetryBundle(
        spacecraft=name,
        channels=channels,
        train=train.astype(np.float32),
        test=test.astype(np.float32),
        train_cmd=train_cmd.astype(np.float32),
        test_cmd=test_cmd.astype(np.float32),
        labels=labels,
        full_test=full_test,
        full_cmd=full_cmd,
        full_train=full_train,
        full_train_cmd=full_train_cmd,
    )


def make_windows(
    values: np.ndarray, cmds: np.ndarray, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows for next-step forecasting.

    Returns (X, y, t) where X is (N, window, C+24), y is (N, C) and t holds the
    index in the original series that each target corresponds to.
    """
    feats = np.concatenate([values, cmds], axis=1)
    n = len(values) - window
    if n <= 0:
        raise ValueError(f"series of length {len(values)} is shorter than window {window}")
    idx = np.arange(n)
    X = np.stack([feats[i : i + window] for i in idx])
    y = values[window:]
    t = idx + window
    return X.astype(np.float32), y.astype(np.float32), t


if __name__ == "__main__":
    for sc in SPACECRAFT:
        b = load_spacecraft(sc)
        n_lab = sum(len(v) for v in b.labels.values())
        print(
            f"{sc:5s} channels={b.n_channels:3d} "
            f"train={b.train.shape} test={b.test.shape} "
            f"labelled_anomaly_windows={n_lab}"
        )
        X, y, t = make_windows(b.test, b.test_cmd)
        print(f"      test windows X={X.shape} y={y.shape}")
