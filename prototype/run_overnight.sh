#!/bin/bash
# Unattended pipeline: waits for the 82-channel retrain, then runs everything
# that depends on it and writes a summary.
#
# Each stage is independent and failures are recorded rather than fatal, so one
# broken step cannot silently discard the stages after it.

cd "C:/Users/Tharun/Documents/SIH Project/prototype/src" || exit 1
LOG="../reports/overnight.log"
: > "$LOG"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
stage() {
  local name="$1"; shift
  say "START $name"
  if "$@" >> "$LOG" 2>&1; then
    say "OK    $name"
  else
    say "FAIL  $name (exit $?) - continuing"
  fi
}

say "waiting for the 82-channel retrain to finish"
# The retrain writes its metadata file last, so its presence means done.
while [ ! -f ../models/msl_channels.json ] || [ ! -f ../models/smap_channels.json ]; do
  sleep 60
done
say "retrain complete: $(ls ../models/smap_channels/*.pt 2>/dev/null | wc -l) SMAP + $(ls ../models/msl_channels/*.pt 2>/dev/null | wc -l) MSL models"

# Detection results are memoised per process, but each stage is its own process,
# so nothing stale carries across.
stage "cross-mission tuning"      python -u tune.py --quiet
stage "evaluation"                python -u evaluate.py --sweep
stage "early-warning measurement" python -u prewarning.py
stage "database rebuild"          python -u populate_db.py --with-readings

say "----- SUMMARY -----"
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
R = pathlib.Path("../reports")

def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None

t = load("tuned_evaluation.json")
if t:
    print("\nHeld-out detection (config chosen on the other mission):")
    for sc in t:
        h = t[sc]["held_out_result"]
        ci = h.get("top1_95ci") or (0, 0)
        print(f"  {sc:5s} precision {h['precision']:.3f}  recall {h['recall']:.3f}"
              f"  F1 {h['f1']:.3f}   labelled windows {h['n_labelled']}")
        print(f"        attribution top-1 {h['top1_channel_accuracy']} "
              f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}] over "
              f"{h['events_matched_to_labels']} events")
        if h.get("operational_precision") is not None:
            print(f"        operational precision {h['operational_precision']:.3f} "
                  f"({h.get('incidents_during_a_real_fault')}/{h.get('n_incidents')})")

p = load("prewarning.json")
if p:
    print("\nEarly warning (lead time before NASA's labelled window opens):")
    for sc, v in p.items():
        for level in ("alert", "watch"):
            s = v[level]
            if s.get("median_lead") is None:
                print(f"  {sc:5s} {level:5s}: never fired before onset")
            else:
                lo, hi = s["rate_95ci"]
                print(f"  {sc:5s} {level:5s}: {s['warned']}/{s['of']} = {s['rate']:.0%} "
                      f"(95% CI {lo:.0%}-{hi:.0%}), median lead "
                      f"{s['median_lead']} readings, max {s['max_lead']}")
PY

say "----- DONE -----"
