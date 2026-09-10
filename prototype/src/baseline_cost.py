"""What does the fixed-limit alarm cost in false alarms?

Speaking first is only a virtue if the alarm is not speaking all the time. This
counts, for the same train-range redline, how many readings outside any labelled
anomaly window trip it.
"""
import warnings, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import WINDOW
from data import load_spacecraft
from detect import detect

for craft in ("SMAP", "MSL"):
    b = load_spacecraft(craft)
    res = detect(b)
    t = res["t"]

    ours_by_ch = {}
    for e in res["events"]:
        ours_by_ch.setdefault(e["channel"], []).append(
            (int(t[e["start"]]), int(t[e["end"]])))

    lim_fp = lim_scored = our_fp = our_scored = 0
    lim_ch_firing = our_ch_firing = n_ch = 0
    for ch, windows in b.labels.items():
        te, tr = b.full_test.get(ch), b.full_train.get(ch)
        if te is None or tr is None:
            continue
        n_ch += 1
        lo, hi = float(np.nanmin(tr)), float(np.nanmax(tr))
        healthy = np.ones(len(te), dtype=bool)
        healthy[:WINDOW] = False
        for (s, e) in windows:
            healthy[max(0, int(s)):int(e) + 1] = False

        breach = ((te > hi) | (te < lo)) & healthy
        lim_fp += int(breach.sum()); lim_scored += int(healthy.sum())
        lim_ch_firing += bool(breach.any())

        ours = np.zeros(len(te), dtype=bool)
        for (s, e) in ours_by_ch.get(ch, []):
            ours[s:e + 1] = True
        of = int((ours & healthy).sum())
        our_fp += of
        our_ch_firing += bool((ours & healthy).any())

    print(f"{craft}: {n_ch} labelled channels, {lim_scored} healthy readings scored")
    print(f"   fixed limit  false alarms on {lim_fp:6d} healthy readings "
          f"({100*lim_fp/max(lim_scored,1):5.2f}%)  on {lim_ch_firing}/{n_ch} channels")
    print(f"   our detector false alarms on {our_fp:6d} healthy readings "
          f"({100*our_fp/max(lim_scored,1):5.2f}%)  on {our_ch_firing}/{n_ch} channels")
    print()
