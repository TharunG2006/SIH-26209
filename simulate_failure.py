import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def simulate_failure(parquet_name="AEPEX_68506.parquet"):
    path = Path(f"prototype/data/satnogs/{parquet_name}")
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    
    # Take the last row as a baseline for the new frames
    last_row = df.iloc[-1:].copy()
    
    # Try to parse the timestamp; fallback to current time
    try:
        ts = pd.to_datetime(last_row['timestamp'].iloc[0])
    except:
        ts = datetime.utcnow()
        
    print(f"Baseline timestamp: {ts}")
    
    # We will generate 30 minutes of fake data
    new_rows = []
    
    for i in range(1, 31):
        row = last_row.copy()
        row['timestamp'] = (ts + timedelta(minutes=i)).isoformat()
        
        # Add slight natural noise to all numeric channels
        for col in row.columns:
            if col not in ('timestamp', 'timestep', 'observation_id') and pd.api.types.is_numeric_dtype(row[col]):
                row[col] += np.random.normal(0, 0.05 * np.abs(row[col].iloc[0]) + 0.01)
                
        # --- CATASTROPHIC FAILURE SIMULATION ---
        # At t=10 minutes, Reaction Wheel 3 jams and its speed drops to zero, 
        # causing a massive current spike as the motor fights the jam
        if i >= 10:
            if 'sw_adcs_wheel_sp3' in row:
                row['sw_adcs_wheel_sp3'] = 0.0  # Wheel stopped
            
        # At t=14 minutes, the satellite loses attitude control 
        # and begins tumbling out of control (body rates spike)
        if i >= 14:
            if 'sw_adcs_body_rt1' in row:
                row['sw_adcs_body_rt1'] = 800.0  # Tumbling on X axis
            if 'sw_adcs_body_rt2' in row:
                row['sw_adcs_body_rt2'] = -600.0 # Tumbling on Y axis
                
        # At t=20, the tumbling causes the solar arrays to lose sun pointing, 
        # so voltage and power plummet
        if i >= 20:
            for col in row.columns:
                if 'voltage' in col.lower() or 'power' in col.lower() or 'eps' in col.lower() or 'psu' in col.lower():
                    if pd.api.types.is_numeric_dtype(row[col]):
                        row[col] = row[col].iloc[0] * 0.1 # 90% power loss
                        
        new_rows.append(row)
        
    print("Injecting 30 minutes of catastrophic failure data...")
    df = pd.concat([df] + new_rows, ignore_index=True)
    
    # Save back to the parquet
    df.to_parquet(path)
    
    # CLEAR THE FORECAST CACHE
    # detect.py caches inference to disk to keep the dashboard fast.
    # We must wipe it so the dashboard is forced to run inference on our fake data.
    cache_dir = Path("prototype/models/forecast_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("aepex_*.npz"):
            f.unlink()
            
    print("\nSUCCESS: The failure has been injected into the live telemetry stream!")
    print("Go to your Streamlit dashboard and click 'Reload Data from Disk' in the sidebar.")
    print("You should see a massive CRITICAL anomaly appear.")

if __name__ == "__main__":
    simulate_failure()
