"""Select detection hyperparameters without cheating on the test labels.

The tempting move is to sweep thresholds on SMAP, keep whichever setting scores
best on SMAP, and report that number.  That is tuning on the test answers, and
a reviewer is entitled to discount the result entirely.

Instead we tune **across missions**: hyperparameters chosen on MSL are used to
report SMAP, and vice versa.  Every reported number is therefore produced by a
configuration selected without ever looking at that spacecraft's labels, which
is the same discipline as a held-out validation set.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

from config import REPORT_DIR, SPACECRAFT
from evaluate import evaluate

# Detection knobs that trade recall against precision.
GRID = {
    "prune_drop": [0.0, 0.02, 0.05, 0.10, 0.13, 0.20],
    "min_run": [3, 5, 10],
    "z_grid_max": [6.0, 10.0],
    # Whether the per-channel threshold is chosen on the raw smoothed error or
    # on its robust z-score.  Which one wins is mission-dependent, so it is
    # selected by the same cross-mission protocol as everything else rather
    # than picked by hand per spacecraft.
    "threshold_signal": ["err", "z", "both"],
}


def _configs():
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, combo))


def score(spacecraft: str, cfg: dict) -> dict:
    """Evaluate one configuration by patching the module-level knobs."""
    import detect

    old = (detect.PRUNE_DROP, detect.Z_SEARCH_MAX, detect.THRESHOLD_SIGNAL)
    detect.PRUNE_DROP = cfg["prune_drop"]
    detect.Z_SEARCH_MAX = cfg["z_grid_max"]
    detect.THRESHOLD_SIGNAL = cfg["threshold_signal"]
    try:
        return evaluate(spacecraft, min_run=cfg["min_run"])
    finally:
        (detect.PRUNE_DROP, detect.Z_SEARCH_MAX,
         detect.THRESHOLD_SIGNAL) = old


def tune(tune_on: str, report_on: str, verbose: bool = True) -> dict:
    """Pick the best-F1 config on `tune_on`, then report it on `report_on`."""
    trials = []
    for cfg in _configs():
        s = score(tune_on, cfg)
        trials.append({**cfg, "f1": s["f1"], "precision": s["precision"],
                       "recall": s["recall"], "n": s["n_predicted"]})
        if verbose:
            print(f"    [{tune_on}] {cfg} -> P={s['precision']:.3f} "
                  f"R={s['recall']:.3f} F1={s['f1']:.3f}", flush=True)

    best = max(trials, key=lambda t: (t["f1"], t["recall"]))
    cfg = {k: best[k] for k in GRID}
    held_out = score(report_on, cfg)

    return {
        "tuned_on": tune_on,
        "reported_on": report_on,
        "chosen_config": cfg,
        "config_score_on_tuning_mission": {
            k: best[k] for k in ("precision", "recall", "f1")
        },
        # Everything except the raw baseline block, so a metric added to
        # evaluate() does not silently vanish from the report.
        "held_out_result": {k: v for k, v in held_out.items()
                            if k != "telemanom_baseline"},
        "n_configs_tried": len(trials),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    names = list(SPACECRAFT)
    report = {}
    for tune_target, report_target in ((names[1], names[0]), (names[0], names[1])):
        print(f"\n=== tune on {tune_target}, report on {report_target} ===")
        r = tune(tune_target, report_target, verbose=not args.quiet)
        report[report_target] = r
        h = r["held_out_result"]
        print(f"  chosen config (from {tune_target}): {r['chosen_config']}")
        print(f"  HELD-OUT {report_target}: precision {h['precision']:.3f}  "
              f"recall {h['recall']:.3f}  F1 {h['f1']:.3f}")
        print(f"  attribution top-1 {h['top1_channel_accuracy']} "
              f"(over {h['events_matched_to_labels']} matched events)")

    (REPORT_DIR / "tuned_evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT_DIR / 'tuned_evaluation.json'}")
