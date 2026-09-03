"""Regression tests for scoring, live-channel selection and factor calibration.

The defects these cover were the dangerous kind: none of them raised an error.
They produced numbers that looked entirely reasonable and were wrong - a "60σ
anomaly" in a network packet header, a factor that explained 100% of events by
firing constantly, anomalies counted as missed in data never loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import WINDOW                                    # noqa: E402
from data import TelemetryBundle                             # noqa: E402
from evaluate import _overlaps, channel_event_scores         # noqa: E402
from live import (_is_protocol, channel_blocks,              # noqa: E402
                  complete_frames, health_channels)


def _bundle(labels, test_len=1000, channels=("A-1",)):
    n = len(channels)
    vals = np.zeros((test_len, n), dtype=np.float32)
    cmds = np.zeros((test_len, 0), dtype=np.float32)
    return TelemetryBundle(
        spacecraft="TEST", channels=list(channels),
        train=vals, test=vals, train_cmd=cmds, test_cmd=cmds,
        labels=labels,
        full_test={c: vals[:, i] for i, c in enumerate(channels)},
        full_cmd={c: cmds for c in channels},
    )


class TestOverlap:
    def test_touching_windows_overlap(self):
        assert _overlaps((10, 20), (20, 30))

    def test_adjacent_windows_do_not(self):
        assert not _overlaps((10, 20), (21, 30))

    def test_containment(self):
        assert _overlaps((10, 100), (40, 50))


class TestScorableLabels:
    def test_anomalies_beyond_the_loaded_data_are_excluded(self):
        # MSL channels run to 6100 readings but were truncated to 2038, so 7 of
        # 26 labelled anomalies sat past the end of the data ever loaded - and
        # were counted as misses, understating recall for a reason unrelated to
        # the detector.
        b = _bundle({"A-1": [(500, 600), (5000, 5100)]}, test_len=1000)
        keep = b.scorable_labels({"A-1": 1000})["A-1"]
        assert keep == [(500, 600)]

    def test_anomalies_inside_the_lead_in_are_excluded(self):
        # Forecasting only begins at WINDOW, so a window wholly inside the
        # lead-in has no prediction to deviate from and can never be detected.
        early = (0, max(1, WINDOW - 10))
        late = (WINDOW + 100, WINDOW + 200)
        b = _bundle({"A-1": [early, late]}, test_len=WINDOW + 500)
        keep = b.scorable_labels({"A-1": WINDOW + 500})["A-1"]
        assert keep == [late]

    def test_dropped_count_matches(self):
        b = _bundle({"A-1": [(500, 600), (5000, 5100)]}, test_len=1000)
        assert b.dropped_label_count({"A-1": 1000}) == 1


class TestChannelEventScores:
    @staticmethod
    def _result(events, channels=("A-1",), length=1000):
        return {
            "events": events,
            "channels": list(channels),
            "t": np.arange(length),
            "channel_lengths": {c: length for c in channels},
        }

    def test_a_hit_on_the_right_channel_counts(self):
        b = _bundle({"A-1": [(400, 500)]})
        r = self._result([{"channel": "A-1", "start": 420, "end": 460}])
        s = channel_event_scores(r, b)
        assert s["precision"] == 1.0 and s["recall"] == 1.0

    def test_right_moment_wrong_channel_is_a_miss_and_a_false_alarm(self):
        # The strict convention is deliberate: it is what makes these figures
        # comparable to telemanom, and it is why MSL scores lower.
        b = _bundle({"A-1": [(400, 500)], "B-1": []}, channels=("A-1", "B-1"))
        r = self._result([{"channel": "B-1", "start": 420, "end": 460}],
                         channels=("A-1", "B-1"))
        s = channel_event_scores(r, b)
        assert s["precision"] == 0.0
        assert s["recall"] == 0.0
        assert s["false_positives"] == 1 and s["false_negatives"] == 1

    def test_no_detections_gives_zero_recall_not_a_crash(self):
        b = _bundle({"A-1": [(400, 500)]})
        s = channel_event_scores(self._result([]), b)
        assert s["recall"] == 0.0 and s["precision"] == 0.0

    def test_several_detections_inside_one_window_all_count(self):
        b = _bundle({"A-1": [(400, 900)]})
        r = self._result([
            {"channel": "A-1", "start": 420, "end": 440},
            {"channel": "A-1", "start": 600, "end": 650},
        ])
        s = channel_event_scores(r, b)
        assert s["true_positives"] == 2 and s["precision"] == 1.0


class TestLiveChannelSelection:
    def test_protocol_headers_are_rejected(self):
        # The first GRBBeta run trained on csp_hdr_source and reported a 60
        # sigma "anomaly" - packet routing metadata, present in 97% of frames,
        # while the real telemetry sits in 12% and was filtered out.
        assert _is_protocol("csp_hdr_source")
        assert _is_protocol("csp_hdr_dst_port")
        assert _is_protocol("ax25_frame")
        assert not _is_protocol("psu_bat_volt")
        assert not _is_protocol("uhf_rf_chip_act_temperature")

    def test_blocks_group_by_subsystem_and_ignore_protocol(self):
        df = pd.DataFrame({
            "csp_hdr_source": np.arange(300) % 7,
            "psu_bat_volt": np.random.rand(300),
            "psu_pv_in_volt1": np.random.rand(300),
            "uhf_rf_chip_act_temperature": np.random.rand(300),
            "timestep": np.arange(300),
        })
        blocks = channel_blocks(df, min_frames=100)
        names = {b["subsystem"] for b in blocks}
        assert "csp" not in names
        assert "psu" in names

    def test_counters_are_not_treated_as_health_telemetry(self):
        # A monotonic counter is near-perfectly forecastable, so it inflates
        # apparent accuracy while saying nothing about the spacecraft.
        n = 300
        df = pd.DataFrame({
            "psu_uptime_tot": np.arange(n, dtype=float),
            "psu_packets_recvd_cnt": np.arange(n, dtype=float) * 2,
            "psu_bat_volt": np.random.default_rng(0).normal(8, 0.4, n),
            "timestep": np.arange(n),
        })
        keep = health_channels(df, min_unique=5, min_coverage=0.5)
        assert "psu_bat_volt" in keep
        assert "psu_uptime_tot" not in keep
        assert "psu_packets_recvd_cnt" not in keep

    def test_complete_frames_keeps_only_rows_carrying_the_block(self):
        # A satellite interleaves frame types, so a field is absent - not zero -
        # in frames of another type. Interpolating would invent telemetry.
        df = pd.DataFrame({
            "psu_bat_volt": [1.0, np.nan, 3.0, np.nan],
            "psu_pv_in_volt1": [2.0, np.nan, 4.0, 5.0],
            "timestep": [0, 1, 2, 3],
        })
        out = complete_frames(df, ["psu_bat_volt", "psu_pv_in_volt1"])
        assert len(out) == 2
        assert list(out["timestep"]) == [0, 1]      # renumbered, not original


class TestFactorCalibration:
    def test_a_factor_firing_everywhere_is_not_discriminative(self):
        # Command bits precede 79% of all SMAP readings, so "a command came
        # before this anomaly" is almost always true and explains nothing.
        import factors
        assert factors.MIN_USEFUL_LIFT >= 2.0
        assert factors.REPORT_COMMANDING is False, (
            "commanding measured lift ~1.0 and must stay off by default")

    def test_early_warning_requires_demonstrated_lift(self):
        import early
        assert early.MIN_USEFUL_LIFT > 1.0
        verdict = {"lift": 1.0, "is_early_warning": False}
        assert not verdict["is_early_warning"]

    def test_cusum_reacts_to_sustained_drift_not_a_single_spike(self):
        import early
        rng = np.random.default_rng(0)
        quiet = rng.normal(1.0, 0.05, 400)

        spike = quiet.copy()
        spike[200] = 20.0                       # one huge reading
        drift = quiet.copy()
        drift[200:] += 0.6                      # small, persistent

        assert early.cusum_alarm(drift).sum() > early.cusum_alarm(spike).sum()
