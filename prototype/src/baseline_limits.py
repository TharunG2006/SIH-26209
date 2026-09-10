"""Who speaks first, counted only over anomalies BOTH alarms actually catch.

A detection counts as catching a window if it overlaps it, which is the rule the
evaluator uses; the alarm time is then the first reading each one flags at or
after the window opens.
"""
import warnings, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import WINDOW
from data import load_spacecraft
from detect import detect

def main() -> None:
    for craft in ("SMAP", "MSL"):
        b = load_spacecraft(craft)
        res = detect(b)
        t = res["t"]
        ours_by_ch = {}
        for e in res["events"]:
            ours_by_ch.setdefault(e["channel"], []).append(
                (int(t[e["start"]]), int(t[e["end"]])))

        we = lim = tie = 0
        both = we_only = lim_only = none = 0
        leads = []
        for ch, windows in b.labels.items():
            te, tr = b.full_test.get(ch), b.full_train.get(ch)
            if te is None or tr is None:
                continue
            lo, hi = float(np.nanmin(tr)), float(np.nanmax(tr))
            for (s, e) in windows:
                s, e = int(s), int(e)
                if e < WINDOW or s >= len(te):
                    continue
                a, z = max(s, WINDOW), min(e, len(te) - 1)
                if a > z:
                    continue
                seg = te[a:z + 1]
                br = np.where((seg > hi) | (seg < lo))[0]
                t_lim = a + int(br[0]) if len(br) else None
                # overlap, not containment
                hits = [max(st, a) for (st, en) in ours_by_ch.get(ch, [])
                        if en >= a and st <= z]
                t_we = min(hits) if hits else None

                if t_we is not None and t_lim is not None:
                    both += 1
                    if t_we < t_lim:   we += 1;  leads.append(t_lim - t_we)
                    elif t_we > t_lim: lim += 1
                    else:              tie += 1
                elif t_we is not None: we_only += 1
                elif t_lim is not None: lim_only += 1
                else: none += 1

        print(f"{craft}:")
        print(f"   both caught it   {both}   -> we first {we}, limit first {lim}, same {tie}")
        print(f"   only we caught it     {we_only}")
        print(f"   only the limit caught {lim_only}")
        print(f"   neither                {none}")
        if leads:
            print(f"   when we are first, median lead {int(np.median(leads))} readings")
        print()



if __name__ == "__main__":
    main()
