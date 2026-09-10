"""Is there a precursor in ESA telemetry, split by the class of anomaly?

NASA's benchmark is 59% point anomalies - instantaneous, nothing to see
beforehand - so a negative early-warning result there says as much about the
labels as about the method. ESA Mission1 is the opposite: 1142 of its 1203
anomalies are annotated Subsequence. If a precursor exists anywhere, it is here.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import esa
import precursor as P

MISSION = sys.argv[1] if len(sys.argv) > 1 else "Mission1"


def run(subset_name: str, rows, ann_all, channels, rng):
    per = []
    for ch in channels:
        chan_rows = rows[rows["Channel"] == ch]
        if chan_rows.empty:
            continue
        try:
            s = esa.load_channel(ch, MISSION)
        except Exception as e:                       # a channel we cannot read
            print(f"    ({ch}: {type(e).__name__})", flush=True)
            continue
        v = s.to_numpy(dtype=float)
        idx = s.index

        onsets = []
        for _, r in chan_rows.iterrows():
            if r.isna()["StartTime"]:
                continue
            onsets.append(int(idx.searchsorted(r["StartTime"])))

        # Everything annotated on this channel is off-limits for the baseline,
        # whatever its category - a rare nominal event is still not "quiet".
        blocked = np.zeros(len(v), dtype=bool)
        for _, r in ann_all[ann_all["Channel"] == ch].iterrows():
            if r.isna()["StartTime"] or r.isna()["EndTime"]:
                continue
            a = max(0, int(idx.searchsorted(r["StartTime"])) - P.MAX_LEAD)
            b = min(len(v), int(idx.searchsorted(r["EndTime"])) + P.MAX_LEAD)
            if b > a:
                blocked[a:b] = True

        per.append(P.test_onsets(v, onsets, blocked, rng))
        print(f"    {ch}: {per[-1][2]} onsets tested", flush=True)
    return P.summarise_counts(per)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ann = esa.load_annotations(MISSION)
    anomalies = ann[ann["Category"] == "Anomaly"]

    print(f"{MISSION}: {len(anomalies)} anomalies "
          f"({(anomalies['Length'] == 'Subsequence').sum()} subsequence, "
          f"{(anomalies['Length'] == 'Point').sum()} point)\n")

    for name, rows in (
        ("SUBSEQUENCE anomalies (extended in time)",
         anomalies[anomalies["Length"] == "Subsequence"]),
        ("POINT anomalies (instantaneous)",
         anomalies[anomalies["Length"] == "Point"]),
    ):
        chans = sorted(rows["Channel"].dropna().unique())
        print(f"=== {name}: {len(rows)} events on {len(chans)} channels ===",
              flush=True)
        table = run(name, rows, ann, chans, np.random.default_rng(0))
        P.print_table(f"{MISSION} - {name}", table)
        print()
