import urllib.request
import json
import streamlit as st

@st.cache_data(ttl=600)  # Cache for 10 minutes to avoid NOAA rate limits
def get_space_weather():
    try:
        # Kp Index (Geomagnetic Storms)
        kp_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        with urllib.request.urlopen(kp_url, timeout=5) as r:
            kp_data = json.loads(r.read())
        
        if isinstance(kp_data[-1], dict):
            kp_val = float(kp_data[-1].get('Kp', 0))
        else:
            kp_val = float(kp_data[-1][1])
            
        if kp_val >= 8:
            storm = "G4/G5 (Severe)"
            kp_color = "#d62728"
        elif kp_val >= 5:
            storm = f"G{int(kp_val - 4)} (Storm)"
            kp_color = "#ff7f0e"
        elif kp_val >= 4:
            storm = "Active"
            kp_color = "#e6b800"
        else:
            storm = "Quiet"
            kp_color = "#2ca02c"
            
        # X-ray Flux (Solar Flares)
        xray_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
        with urllib.request.urlopen(xray_url, timeout=5) as r:
            xray_data = json.loads(r.read())
        
        flux = float(xray_data[-1].get('flux', 0))
        
        if flux >= 1e-4:
            flare = "X-Class (Extreme)"
            flare_color = "#d62728"
        elif flux >= 1e-5:
            flare = "M-Class (Moderate)"
            flare_color = "#ff7f0e"
        elif flux >= 1e-6:
            flare = "C-Class (Minor)"
            flare_color = "#e6b800"
        else:
            flare = "Normal"
            flare_color = "#2ca02c"
            
        return {
            "kp_val": kp_val,
            "storm_status": storm,
            "kp_color": kp_color,
            "flux_val": flux,
            "flare_status": flare,
            "flare_color": flare_color,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
