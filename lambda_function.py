import json
import os
import boto3

# Add prototype/src to path so we can import from the existing codebase
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "prototype" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live import detect_live, load_frames
from satnogs import refresh_capture

def handler(event, context):
    """AWS Lambda entry point for serverless anomaly detection."""
    print("Lambda triggered. Event:", event)
    norad_id = event.get('norad_id', 43013)  # Default NOA-14
    s3_bucket = os.environ.get("S3_BUCKET")
    sns_arn = os.environ.get("SNS_TOPIC_ARN")
    
    # 1. Fetch live telemetry from SatNOGS (1 hour)
    print(f"Fetching latest frames for NORAD {norad_id}...")
    res = refresh_capture(norad_id, hours=1)
    
    if not res.get("new_frames") or not res.get("path"):
        return {"statusCode": 200, "body": "No new telemetry data to process."}
        
    path = res["path"]
    
    # 2. Upload raw data to S3 Data Lake
    if s3_bucket:
        try:
            s3 = boto3.client('s3')
            s3.upload_file(str(path), s3_bucket, f"telemetry/latest_{norad_id}.parquet")
            print(f"Uploaded to s3://{s3_bucket}/telemetry/latest_{norad_id}.parquet")
        except Exception as e:
            print(f"S3 Upload failed: {e}")
            
    # 3. Run PyTorch Inference
    print("Running AI Inference...")
    df = load_frames(path)
    result = detect_live(df, list(df.columns), str(norad_id))
    
    print(f"Detection complete. Found {len(result['anomalies'])} anomalies.")
    
    # 4. Trigger SNS if CRITICAL
    if result["anomalies"] and sns_arn:
        peak_anomaly = max(result["anomalies"], key=lambda x: x.severity)
        if peak_anomaly.severity >= 8.0:
            print(f"CRITICAL ANOMALY DETECTED! Dispatching SNS to {sns_arn}...")
            try:
                sns = boto3.client('sns', region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-north-1"))
                sns.publish(
                    TopicArn=sns_arn,
                    Message=f"CRITICAL ANOMALY on satellite {norad_id}!\n\nDuration: {peak_anomaly.duration} timesteps\nPeak severity: {peak_anomaly.severity:.1f}x normal\n\nExplanation: {peak_anomaly.explanation()}",
                    Subject=f"CRITICAL: Spacecraft Alert {norad_id}"
                )
            except Exception as e:
                print(f"SNS Publish failed: {e}")

    return {
        'statusCode': 200,
        'body': json.dumps(f"Processed {len(df)} frames. Found {len(result['anomalies'])} anomalies.")
    }
