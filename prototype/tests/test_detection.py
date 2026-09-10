"""Regression tests for the detection and scoring core.

Every test here corresponds to a defect that actually occurred in this project.
They are written against the specific wrong behaviour rather than as generic
coverage, because the failures that mattered were not crashes - they produced
plausible numbers that were wrong, which is exactly what a test catches and a
reading of the code does not.

Run with:  python -m pytest prototype/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import detect as D                      # noqa: E402
from config import SPACECRAFT, WINDOW    # noqa: E402
from data import make_windows            # noqa: E402
from evaluate import wilson_interval     # noqa: E402


# --------------------------------------------------------------- sequences --
class TestFindSequences:
    """min_run must count consecutive flagged points, not the merged span."""

    def test_isolated_blips_do_not_pass_min_run(self):
        # Two single flagged readings 40 apart were merged into one 41-step
        # sequence that satisfied min_run=5, so the noise filter never fired
        # and the operator saw a long anomaly built from two samples.
        flag = np.zeros(200, dtype=bool)
        flag[10] = flag[50] = True
        assert D.find_sequences(flag, min_run=5, merge_gap=50) == []

    def test_a_genuine_run_survives_merging(self):
        flag = np.zeros(200, dtype=bool)
        flag[10:18] = True          # 8 consecutive: real
        flag[50:53] = True          # 3 more, merged in by the gap
        seqs = D.find_sequences(flag, min_run=5, merge_gap=50)
        assert seqs == [(10, 52)]

    def test_gap_wider_than_merge_gap_splits(self):
        flag = np.zeros(400, dtype=bool)
        flag[10:20] = True
        flag[300:310] = True
        assert len(D.find_sequences(flag, min_run=5, merge_gap=50)) == 2

    def test_empty_input(self):
        assert D.find_sequences(np.zeros(50, dtype=bool), 3, 10) == []


class TestPruneSequences:
    """Pruning keeps everything above the *last* significant drop."""

    def test_equally_strong_sequences_all_survive(self):
        # Breaking at the first small drop discarded every weaker sequence
        # below it, which collapsed recall from 0.38 to 0.11.
        e = np.zeros(300)
        for start in (10, 100, 200):
            e[start:start + 10] = 10.0
        seqs = [(10, 19), (100, 109), (200, 209)]
        kept = D.prune_sequences(seqs, e, eps=1.0, min_drop=0.13)
        assert len(kept) == 3

    def test_a_lone_strong_sequence_is_kept(self):
        e = np.zeros(300)
        e[100:110] = 50.0
        kept = D.prune_sequences([(100, 109)], e, eps=1.0, min_drop=0.13)
        assert kept == [(100, 109)]

    def test_empty(self):
        assert D.prune_sequences([], np.zeros(10), eps=1.0) == []


# ------------------------------------------------------------ error scoring --
class TestErrorScores:
    def test_near_zero_mad_does_not_explode(self):
        # A well-forecast channel had an error MAD near zero, so any deviation
        # divided out to ~250,000 sigma and that one channel "explained" every
        # event in the run.
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, (500, 1)).astype(np.float32)
        y_pred = y_true.copy()
        y_pred[250] += 0.5                      # one small deviation
        _, z = D.error_scores(y_true, y_pred)
        assert np.nanmax(z) < 1000, "scale floor is not holding"

    def test_nan_padding_is_confined_to_its_channel(self):
        # Channels run at their own lengths, so a shorter one is NaN-padded.
        # Those NaNs must not drag another channel's median or spread around.
        y_true = np.random.rand(300, 2).astype(np.float32)
        y_pred = y_true.copy()
        y_pred[:, 1] = np.nan
        err, z = D.error_scores(y_true, y_pred)
        assert np.isnan(z[:, 1]).all()
        assert not np.isnan(z[:, 0]).any()

    def test_a_real_deviation_still_scores_high(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(0, 1, (500, 1)).astype(np.float32)
        y_pred = y_true + rng.normal(0, 0.05, y_true.shape).astype(np.float32)
        y_true[300:320] += 8.0                  # a genuine excursion
        _, z = D.error_scores(y_true, y_pred)
        assert np.nanmax(z[295:330]) > 3.0


class TestDynamicThreshold:
    def test_penalty_is_linear_in_points_quadratic_in_sequences(self):
        # Transposing these terms made the threshold so conservative that a
        # whole spacecraft returned zero detections.
        e = np.concatenate([np.full(400, 0.1), np.full(20, 5.0)])
        eps = D.dynamic_threshold(e, min_run=3)
        assert 0.1 < eps < 5.0, "threshold should separate the excursion"

    def test_flat_signal_yields_an_unreachable_threshold(self):
        e = np.full(200, 0.5)
        assert D.dynamic_threshold(e, min_run=3) > 0.5


# ----------------------------------------------------------------- windows --
class TestMakeWindows:
    def test_returns_a_view_not_a_copy(self):
        # Materialising every window allocated ~400 MB and failed outright
        # when the machine was also training models.
        values = np.random.rand(500, 3).astype(np.float32)
        cmds = np.zeros((500, 2), dtype=np.float32)
        X, _, _ = make_windows(values, cmds, window=100)
        assert X.flags.owndata is False

    def test_window_aligns_with_its_target(self):
        # Window i must cover readings [i, i+window) and predict i+window.
        values = np.arange(60, dtype=np.float32).reshape(-1, 1)
        cmds = np.zeros((60, 0), dtype=np.float32)
        X, y, t = make_windows(values, cmds, window=10)
        assert X[0, 0, 0] == 0.0 and X[0, -1, 0] == 9.0
        assert y[0, 0] == 10.0
        assert t[0] == 10

    def test_series_shorter_than_window_raises(self):
        with pytest.raises(ValueError):
            make_windows(np.zeros((5, 1), np.float32),
                         np.zeros((5, 0), np.float32), window=10)


# -------------------------------------------------------------- attribution --
class TestAttribution:
    @staticmethod
    def _group():
        return [
            {"channel": "A-1", "index": 0, "start": 100, "end": 120,
             "peak": 110, "z": 30.0},
            {"channel": "B-1", "index": 1, "start": 140, "end": 160,
             "peak": 150, "z": 10.0},
        ]

    def test_shares_sum_to_one_hundred(self):
        y = np.zeros((200, 2), dtype=np.float32)
        contribs, _ = D.attribute(self._group(), np.ones((200, 2)), y, y)
        assert sum(c["share_pct"] for c in contribs) == pytest.approx(100.0)

    def test_ranked_by_magnitude_but_chained_by_onset(self):
        y = np.zeros((200, 2), dtype=np.float32)
        contribs, _ = D.attribute(self._group(), np.ones((200, 2)), y, y)
        assert contribs[0]["channel"] == "A-1"          # largest first
        chain = sorted(contribs, key=lambda c: c["chain_rank"])
        assert chain[0]["channel"] == "A-1"             # earliest first
        assert chain[1]["lag"] == 40

    def test_the_loudest_channel_need_not_be_the_first(self):
        # This distinction is the point of the causal ordering: on SMAP the
        # loudest channel was 511 sigma while a quieter one moved first.
        group = [
            {"channel": "LOUD", "index": 0, "start": 150, "end": 170,
             "peak": 160, "z": 500.0},
            {"channel": "FIRST", "index": 1, "start": 100, "end": 130,
             "peak": 110, "z": 5.0},
        ]
        y = np.zeros((200, 2), dtype=np.float32)
        contribs, _ = D.attribute(group, np.ones((200, 2)), y, y)
        assert contribs[0]["channel"] == "LOUD"
        assert min(contribs, key=lambda c: c["chain_rank"])["channel"] == "FIRST"


# ------------------------------------------------------------------- config --
class TestConfig:
    def test_no_duplicate_channels(self):
        # labeled_anomalies.csv lists P-2 twice. A duplicate is forecast twice
        # and counted twice in both detections and false alarms, and it makes
        # the expected model count permanently exceed the achievable one.
        for name, spec in SPACECRAFT.items():
            chans = spec["channels"]
            assert len(chans) == len(set(chans)), f"{name} has duplicates"

    def test_aligned_is_a_subset_of_channels(self):
        for name, spec in SPACECRAFT.items():
            assert set(spec.get("aligned", [])) <= set(spec["channels"])


# ------------------------------------------------------------- calibration --
class TestWilsonInterval:
    def test_bounds_stay_within_zero_and_one(self):
        # The normal approximation returns bounds above 1 for small n near 1,
        # which is precisely the regime the attribution figures live in.
        lo, hi = wilson_interval(10, 10)
        assert 0.0 <= lo <= hi <= 1.0

    def test_smaller_samples_give_wider_intervals(self):
        lo_small, hi_small = wilson_interval(9, 10)
        lo_big, hi_big = wilson_interval(90, 100)
        assert (hi_small - lo_small) > (hi_big - lo_big)

    def test_zero_trials(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)


# --- what a channel measures, and what counts as health ---------------------

def test_gps_position_names_the_receiver_not_attitude_control():
    """`bcn_adcs_gps_pos_ecef_1` is a GPS reading, not an orientation.

    These names run general to specific, and taking the first recognised token
    labelled the channel "Attitude control position" - which reads as though the
    spacecraft were reporting which way it was facing rather than where it was.
    """
    import sensors
    assert sensors.subsystem_of("bcn_adcs_gps_pos_ecef_1") == "Navigation"
    assert sensors.subsystem_of("bcn_adcs_bod_rt_1") == "Attitude control"
    assert sensors.subsystem_of("bcn_psu_bat_temp_kelvin") == "Power supply"


def test_orbital_geometry_is_not_a_health_channel():
    """Where the spacecraft is says nothing about whether it is healthy.

    A GPS position component swings between large positive and negative values
    every orbit while the distance from Earth's centre stays constant. Sampled
    only during ground-station passes, no forecaster can track that swing, so it
    is reported as a fault - COSMO's only "anomaly" was the satellite orbiting.
    """
    import numpy as np
    import pandas as pd

    import live

    n = 400
    rng = np.random.default_rng(0)
    phase = np.linspace(0, 8 * np.pi, n)
    df = pd.DataFrame({
        "bcn_adcs_gps_pos_ecef_1": 3.4e8 * np.cos(phase),
        "bcn_adcs_gps_pos_ecef_2": 3.4e8 * np.sin(phase),
        "bcn_adcs_bod_rt_1": rng.normal(0, 1, n),
        "bcn_store_part_wr_hk": np.arange(n) + rng.integers(0, 2, n),
    })
    keep = live.health_channels(df, min_unique=4, min_coverage=0.5)
    assert not [c for c in keep if "ecef" in c], "orbital position was kept"
    assert not [c for c in keep if "store_part" in c], "a storage pointer was kept"
    assert "bcn_adcs_bod_rt_1" in keep, "a real health channel was dropped"


def test_redundant_siblings_share_a_unit():
    """Three identical solar panels cannot report in three different units.

    Resolving each channel against its own magnitude put the quietest panel in a
    different bracket from the other two, and the dashboard showed one in watts
    beside two in milliwatts.
    """
    import numpy as np

    import sensors

    values = {
        "psu_pv_in_power1": np.full(50, 4000.0),
        "psu_pv_in_power2": np.full(50, 3800.0),
        "psu_pv_in_power3": np.full(50, 400.0),   # quiet panel, same unit
    }
    units = {d["unit"] for d in
             sensors.describe_group(list(values), values).values()}
    assert units == {"mW"}, f"siblings disagreed on the unit: {units}"


def test_index_digits_do_not_defeat_quantity_matching():
    """`psu_pv_in_amp2` is a current; a word boundary never matched it."""
    import sensors
    assert sensors.quantity_of("psu_pv_in_amp2") == "current"
    assert sensors.quantity_of("psu_pv_in_power3") == "power"
    assert sensors.quantity_of("bcn_adcs_rw_sp_1") == "rotation rate"
    # ...but a bare digit suffix must not turn unrelated names into readings.
    assert sensors.quantity_of("csp_hdr_source") is None


def test_live_manifest_resolves_from_any_directory(tmp_path, monkeypatch):
    """A capture path recorded as typed only resolves from the training shell.

    COSMO was retrained from `prototype/src`, so its manifest read
    `../data/satnogs/COSMO_68460.parquet`; the dashboard runs from
    `prototype/` and could not open it.
    """
    import json

    import sources

    cap = tmp_path / "captures"
    cap.mkdir()
    (cap / "DEMO_1.parquet").write_bytes(b"")
    monkeypatch.setattr(sources, "_live_meta", lambda: {
        "DEMO": {"channels": ["a"], "parquet": "../nowhere/DEMO_1.parquet"}})
    monkeypatch.setattr("satnogs.SATNOGS_DIR", cap)

    # The recorded path is unreachable, so the capture directory must be tried;
    # reaching pandas at all proves the fallback resolved (the file is empty).
    try:
        sources.load_source("DEMO")
    except FileNotFoundError as e:
        raise AssertionError(f"fell back to nothing: {e}") from None
    except Exception:
        pass    # any parse error means the path was found and opened
