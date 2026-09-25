"""Automated pipeline to refresh live SatNOGS telemetry in the background."""

import time
import json
import re
import os
import traceback
from pathlib import Path
import sys

# Ensure we can import from src
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import MODEL_DIR, REPORT_DIR
from satnogs import refresh_capture
from live import load_frames, complete_frames, detect_live

def run_pipeline():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting pipeline refresh...")
    for meta_file in MODEL_DIR.glob("*_channels.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            name = meta.get("satellite")
            
            # Extract NORAD ID from the parquet filename (e.g. COSMO_68460.parquet)
            parquet_path = meta.get("parquet", "")
            m = re.search(r"_(\d+)\.parquet$", parquet_path)
            norad_id = int(m.group(1)) if m else None
            
            if not name or not norad_id:
                continue

            print(f"\n--- Refreshing {name} (NORAD {norad_id}) ---")
            # 1. Fetch new frames (defaults to looking at the last 12 hours of passes)
            res = refresh_capture(norad_id, hours=12, verbose=True)
            if res.get("new_frames", 0) == 0:
                print(f"No new frames for {name}.")
                continue
                
            # 2. Upload to Amazon S3 Data Lake
            path = res["path"]
            s3_bucket = os.environ.get("S3_BUCKET")
            if s3_bucket:
                print(f"Uploading telemetry to S3 Data Lake (s3://{s3_bucket}/telemetry/{path.name})...")
                try:
                    import boto3
                    s3 = boto3.client('s3')
                    s3.upload_file(str(path), s3_bucket, f"telemetry/{path.name}")
                    print("S3 upload successful.")
                except Exception as e:
                    print(f"S3 upload skipped/failed: {e}")

            # 3. Re-detect anomalies on the updated parquet
            df = load_frames(path)
            
            # The chosen block's channels that were previously trained
            chans = list(meta["channels"].keys())
            
            # Filter the dataframe to only include complete frames for this block
            df = complete_frames(df, chans)
            
            print(f"Running detection on {len(df)} frames...")
            result = detect_live(df, chans, name)
            
            # 3. Save report
            ts = df["timestamp"].to_numpy()
            out_file = REPORT_DIR / f"live_{name.lower()}.json"
            out_file.write_text(json.dumps({
                "satellite": name, "frames": len(df), "channels": chans,
                "incidents": [{"start": a.start, "end": a.end,
                               "severity": a.severity,
                               "timestamp": str(ts[min(a.start, len(ts) - 1)]),
                               "explanation": a.explanation(),
                               "propagation": a.propagation()}
                              for a in result["anomalies"]],
            }, indent=2))
            
            print(f"Update complete for {name}. Found {len(result['anomalies'])} incidents.")
            
        except Exception as e:
            print(f"Error refreshing {meta_file.name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60, help="Minutes to wait between refreshes")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    args = ap.parse_args()
    
    if args.once:
        run_pipeline()
    else:
        print(f"Pipeline started. Will check for new telemetry every {args.interval} minutes.")
        while True:
            run_pipeline()
            print(f"\nSleeping for {args.interval} minutes...")
            time.sleep(args.interval * 60)
