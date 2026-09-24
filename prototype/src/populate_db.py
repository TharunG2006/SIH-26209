"""Run detection and persist the results into the seven-table schema.

    python src/populate_db.py                 # anomalies + explanations
    python src/populate_db.py --with-readings # also store raw telemetry
    python src/populate_db.py --show 1        # read one anomaly back out
"""
from __future__ import annotations

import argparse
import json
import sys

import db
from config import REPORT_DIR
from detect import detect
from sources import get_source, list_sources, load_source

# NORAD catalogue numbers for the two benchmark spacecraft.  MSL is the cruise
# stage's catalogue entry; the rover itself is on Mars and has no orbital
# element set, which is why the schema allows norad_id to be NULL.
CATALOGUE = {
    "SMAP": {"norad_id": 40376, "launch_date": "2015-01-31"},
    "MSL": {"norad_id": 37936, "launch_date": "2011-11-26"},
}


def tuned_config(spacecraft: str) -> dict:
    """Reuse the held-out configuration recorded by tune.py, if present."""
    path = REPORT_DIR / "tuned_evaluation.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    return blob.get(spacecraft, {}).get("chosen_config", {})


def populate(spacecraft: str, with_readings: bool = False,
             verbose: bool = True) -> dict:
    source = get_source(spacecraft)
    if source is None:
        raise KeyError(f"unknown source {spacecraft!r}")
    # A live capture has no labelled anomalies, so the held-out configuration
    # tuned on the benchmark does not apply to it; it runs on the defaults.
    cfg = tuned_config(spacecraft) if source.has_labels else {}
    if cfg:
        detect.__globals__["PRUNE_DROP"] = cfg.get("prune_drop",
                                                   detect.__globals__["PRUNE_DROP"])
        detect.__globals__["Z_SEARCH_MAX"] = cfg.get("z_grid_max",
                                                     detect.__globals__["Z_SEARCH_MAX"])
        detect.__globals__["THRESHOLD_SIGNAL"] = cfg.get(
            "threshold_signal", detect.__globals__["THRESHOLD_SIGNAL"])

    bundle, stamps = load_source(spacecraft)
    result = detect(bundle, min_run=cfg.get("min_run", 5 if source.has_labels else 3))

    meta = CATALOGUE.get(spacecraft, {})
    with db.connect() as conn:
        sat_id = db.upsert_satellite(
            conn, spacecraft,
            meta.get("norad_id") or source.norad_id,
            meta.get("launch_date"))
        channel_ids = db.upsert_channels(conn, sat_id, result["channels"])
        n_read = 0
        if with_readings:
            series = {ch: bundle.full_test[ch] for ch in result["channels"]}
            # Live frames carry a real UTC clock; the benchmark ships only
            # reading indices, which is why the schema keeps both columns.
            n_read = db.store_readings(conn, channel_ids, series, stamps=stamps)
        n_anom = db.store_anomalies(conn, sat_id, channel_ids,
                                    result["anomalies"], stamps=stamps)

    out = {"spacecraft": spacecraft, "kind": source.kind,
           "channels": len(channel_ids), "anomalies_written": n_anom,
           "readings_written": n_read, "config": cfg or "defaults"}
    if verbose:
        print(f"[{spacecraft}] {source.kind}: {out['channels']} channels, "
              f"{n_anom} anomalies, {n_read:,} readings")
    return out


def show(anomaly_id: int) -> None:
    with db.connect() as conn:
        rows = db.explain(conn, anomaly_id)
        if not rows:
            print(f"no anomaly with id {anomaly_id}")
            return
        head = rows[0]
        print(f"Anomaly #{head['anomaly_id']} on {head['satellite']}  "
              f"t={head['detected_at']}-{head['ended_at']}  "
              f"severity {head['severity_score']:.1f}  status={head['status']}")
        print(f"\n{'channel':10s} {'subsystem':16s} {'share':>7s} {'z':>9s} "
              f"{'onset':>7s} {'lag':>5s}")
        for r in rows:
            print(f"{r['channel']:10s} {r['subsystem']:16s} "
                  f"{r['share_pct']:6.1f}% {r['deviation_score']:9.1f} "
                  f"{r['onset']:7d} {r['lag']:5d}")
        chain = db.causal_chain(conn, anomaly_id)
        if len(chain) > 1:
            first = chain[0]
            print(f"\nMoved first: {first['channel']} ({first['subsystem']})")
            print("Then: " + ", ".join(
                f"{r['channel']} +{r['lag'] - first['lag']}" for r in chain[1:4]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", default="all",
                    help="a source key from `python src/sources.py`, "
                         "'all', or 'live' for just the SatNOGS captures")
    ap.add_argument("--with-readings", action="store_true",
                    help="also persist raw telemetry (slower, ~250k rows)")
    ap.add_argument("--show", type=int, metavar="ANOMALY_ID",
                    help="print one anomaly and its explanation, then exit")
    ap.add_argument("--alerts", action="store_true",
                    help="print the open alert queue")
    args = ap.parse_args()

    db.init_db()
    if args.show is not None:
        show(args.show)
        raise SystemExit(0)
    if args.alerts:
        with db.connect() as conn:
            for r in db.open_alerts(conn):
                print(f"#{r['anomaly_id']:3d} {r['satellite']:5s} "
                      f"t={r['detected_at']:6d} {r['severity_score']:8.1f}σ  "
                      f"first={r['first_mover']:6s} largest={r['largest_contributor']}")
        raise SystemExit(0)

    if args.spacecraft == "all":
        targets = [s.key for s in list_sources()]
    elif args.spacecraft == "live":
        targets = [s.key for s in list_sources() if s.is_live]
    else:
        targets = [args.spacecraft]
    for sc in targets:
        populate(sc, with_readings=args.with_readings)

    with db.connect() as conn:
        print("\ndatabase summary:")
        for k, v in db.summary(conn).items():
            print(f"  {k:20s} {v:,}")
        print(f"\n{db.DB_PATH}")
