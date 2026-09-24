"""Operator dashboard: replay satellite telemetry and explain every anomaly.

Run with:  streamlit run prototype/dashboard/app.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# torch's CUDA probe imports the deprecated pynvml on import and warns even on a
# CPU-only build.  Harmless, but it prints over the dashboard banner at startup.
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (MIN_RUN, SPACECRAFT, Z_MIN, channel_label,  # type: ignore # noqa: E402
                    has_operator_names)
from detect import detect  # type: ignore # noqa: E402
import sensors  # type: ignore # noqa: E402
from sources import list_sources, load_source, utc_for  # type: ignore # noqa: E402
from space_weather import get_space_weather  # type: ignore # noqa: E402

st.set_page_config(page_title="Satellite Health Monitor", layout="wide",
                   page_icon="🛰️")

# Custom CSS for a professional Mission Control UI
st.markdown("""
<style>
/* Hide the default Streamlit header, footer, and menu for a clean app look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Make metric cards pop with a dark aesthetic */
div[data-testid="metric-container"] {
    background-color: rgba(30, 34, 45, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(243, 156, 18, 0.15);
    border-color: rgba(243, 156, 18, 0.3);
}

/* Custom header styling */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    color: #ffffff !important;
    text-shadow: 0px 0px 15px rgba(243, 156, 18, 0.3);
}

/* Custom subtle scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent; 
}
::-webkit-scrollbar-thumb {
    background: #333; 
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #555; 
}
</style>
""", unsafe_allow_html=True)

# The severity band an alert must reach to count as critical. Also the fixed
# ceiling of the deviation heatmap, so colour is comparable across satellites.
CRITICAL_SIGMA = 8.0

SEVERITY_BANDS = [
    (8.0, "CRITICAL", "#d62728"),
    (5.0, "HIGH", "#ff7f0e"),
    (3.5, "MEDIUM", "#e6b800"),
    (0.0, "LOW", "#2ca02c"),
]


def severity_band(z: float) -> tuple[str, str]:
    for threshold, name, colour in SEVERITY_BANDS:
        if z >= threshold:
            return name, colour
    return "LOW", "#2ca02c"


@st.cache_resource(show_spinner="Loading telemetry and model…")
def load(spacecraft: str, z_min: float | None, min_run: int):
    bundle, stamps = load_source(spacecraft)
    result = detect(bundle, z_min=z_min, min_run=min_run)
    return bundle, result, stamps


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("🛰️ Mission Control")
SOURCES = {s.key: s for s in list_sources()}


def _cached_first(keys: list[str]) -> int:
    """Index of the first source whose forecast is already on disk.

    Forecasting a spacecraft takes minutes; the result is cached, but the
    dropdown used to default to whichever source happened to be listed first.
    If that one was uncached the dashboard sat computing it before the user
    could pick anything else - so the default is a source that opens instantly.
    """
    import detect as _D  # type: ignore

    for i, k in enumerate(keys):
        if _D._disk_path(k, None).exists():
            return i
    return 0


_keys = list(SOURCES)
spacecraft = st.sidebar.selectbox(
    "Satellite", _keys,
    index=_cached_first(_keys),
    format_func=lambda k: SOURCES[k].label,
    help="Benchmark missions carry NASA's labelled anomalies; live captures "
         "carry real channel names and UTC timestamps but no ground truth.",
)
source = SOURCES[spacecraft]
threshold_mode = st.sidebar.radio(
    "Threshold", ["Automatic (recommended)", "Manual Threshold"],
    help="Automatic gives each channel its own nonparametric threshold and "
         "prunes weak sequences — this is the mode the reported metrics use.",
)
if threshold_mode.startswith("Automatic"):
    z_min = None
    st.sidebar.caption("Each channel's threshold is chosen from its own error "
                       "distribution, so a noisy channel cannot drown out a "
                       "quiet one.")
else:
    z_min = st.sidebar.slider("Detection threshold (x normal error)", 1.5, 8.0, float(Z_MIN),
                              0.1, help="Higher = fewer, more confident alerts")
min_run = st.sidebar.slider("Min. duration (timesteps)", 1, 20, MIN_RUN)

st.sidebar.divider()
st.sidebar.subheader("Live Space Weather")
sw = get_space_weather()
if sw.get("success"):
    st.sidebar.markdown(f"**Solar Flares:** <span style='color:{sw['flare_color']}'>{sw['flare_status']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Geomagnetic (Kp):** <span style='color:{sw['kp_color']}'>{sw['storm_status']} (Kp {sw['kp_val']})</span>", unsafe_allow_html=True)
else:
    st.sidebar.caption("NOAA API unavailable.")

with st.sidebar.expander("🧠 Math: AI vs Environmental Shocks"):
    st.markdown('''
**How do we prove it's a fault and not a Solar Flare?**

We use **Random Matrix Theory (RMT)** to isolate internal faults from external shocks (like space weather).

When a satellite operates normally, the correlation matrix $\mathbf{C}$ of its sensors contains purely random noise. The eigenvalues $\lambda$ of this matrix perfectly follow the **Marchenko-Pastur (MP) Distribution**:
''')
    st.latex(r'''\rho(\lambda) = \frac{1}{2\pi \sigma^2} \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{q \lambda}''')
    st.markdown('''
**The Solution:**
1. **Solar Flares / Eclipses:** Shift the *entire* distribution, moving the bulk of the eigenvalues. Astrovia calculates the center of mass of the eigenvalues. If the whole system shifts, the AI classifies it as a benign external environmental shock.
2. **True Hardware Faults:** Cause a single dominant eigenvalue $\lambda_{max}$ to "detach" and spike beyond the upper bound $\lambda_+$, while the rest of the satellite stays in the MP distribution. This mathematically proves a localized, critical internal fault.
''')

if st.sidebar.button("🔄 Reload Data from Disk", use_container_width=True):
    load.clear()
    st.rerun()

bundle, result, stamps = load(spacecraft, z_min, min_run)
channels = result["channels"]


@st.cache_data(show_spinner=False)
def channel_meaning(key: str, names: tuple[str, ...]) -> dict:
    """What each channel measures, with units resolved from its own values. (Cache invalidated 2)

    Channels are passed as separate arrays rather than a shared DataFrame: they
    run to different lengths once each is forecast at its full extent - 4,453 to
    8,640 readings on SMAP - so building one frame from them fails outright.
    Redundant siblings are still resolved together, so three identical solar
    panels cannot end up labelled in three different units.
    """
    b, _ = load_source(key)
    return sensors.describe_group(names, {c: b.full_test.get(c) for c in names
                                          if b.full_test.get(c) is not None})


MEANING = channel_meaning(spacecraft, tuple(channels))


def pretty(ch: str) -> str:
    """Readable name for a channel: what it measures, not just its id."""
    d = MEANING.get(ch, {})
    if d.get("quantity"):
        unit = f" ({d['unit']})" if d.get("unit") else ""
        return f"{d['label']}{unit}"
    return channel_label(ch)
t_axis = result["t"]
n_steps = len(t_axis)

st.sidebar.divider()
mode = st.sidebar.radio("View", ["Live replay", "Full timeline", "Anomaly log"])

# ------------------------------------------------------------------ header ---
st.title("Explainable Satellite Anomaly Detection")
st.caption(
    f"{source.label} — {len(channels)} telemetry channels, "
    f"{n_steps:,} scored readings, forecast by the {result['forecaster']}. "
    "Anomalies are attributed back to the channels that caused them, and "
    "ordered by which channel deviated first."
)

if source.is_live:
    span = ""
    if stamps is not None and len(stamps):
        span = f"  Capture spans {utc_for(stamps, 0)} to {utc_for(stamps, -1)}."
    st.warning(
        "**Live capture — no ground truth.** These are real frames from a "
        "satellite currently in orbit, received by volunteer ground stations. "
        "Nobody publishes when this spacecraft actually malfunctioned, so "
        "nothing here can be scored: it shows the system running on real "
        "telemetry, not how accurate it is. The measured accuracy figures come "
        f"from the NASA benchmark missions.{span}"
    )

_known = sum(1 for d in MEANING.values() if d.get("quantity"))
if _known:
    st.caption(
        f":grey[{_known} of {len(channels)} channels identified by what they "
        "measure. The decoders publish field names but no units, so the "
        "quantity is read from the naming convention and the unit resolved "
        "against the observed magnitudes — inferred, not published metadata.]"
    )

if source.has_labels and not has_operator_names():
    st.caption(
        ":grey[NASA anonymises these channel ids, so channels are shown by id. "
        "Drop a `config/channel_names.json` in (see the example file) to display "
        "your mission's own channel and subsystem names.]"
    )

anomalies = result["anomalies"]
counts = {name: 0 for _, name, _ in SEVERITY_BANDS}
for a in anomalies:
    counts[severity_band(a.severity)[0]] += 1

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Channels monitored", len(channels))
c2.metric("Anomalies detected", len(anomalies))
c3.metric("Critical", counts["CRITICAL"])
c4.metric("High", counts["HIGH"])
if source.has_labels:
    c5.metric("Ground-truth windows",
              sum(len(v) for v in bundle.labels.values()))
else:
    c5.metric("Readings captured", f"{n_steps:,}")


def explanation_panel(anom, key_prefix: str = "") -> None:
    """Render the explainability card for one anomaly."""
    band, colour = severity_band(anom.severity)
    
    # Build human-readable explanation instead of raw mathematical terms
    if anom.contributions:
        parts = []
        for c in anom.contributions[:3]:
            pretty_name = pretty(c["channel"])
            parts.append(f"{pretty_name} ({c['share_pct']:.0f}% contribution)")
        lead = pretty(anom.subsystems[0]["subsystem"]) if anom.subsystems else "unknown"
        explanation = f"Driven by **{lead}**: {', '.join(parts)}."
    else:
        explanation = "No channel attribution available."

    st.markdown(
        f"<div style='border-left:6px solid {colour};padding:0.6rem 1rem;"
        f"background:rgba(128,128,128,0.08);border-radius:4px'>"
        f"<b style='color:{colour}'>{band}</b> &nbsp; "
        f"Duration: {anom.duration} timesteps (t={anom.start:,} to {anom.end:,}) &nbsp;·&nbsp; "
        f"Peak Error: {anom.severity:.1f}x normal threshold at t={anom.peak:,}<br>"
        f"<span style='font-size:0.95em'>{explanation}</span></div>",
        unsafe_allow_html=True,
    )
    # A live capture has a real clock; the benchmark ships only reading indices.
    when = utc_for(stamps, anom.start)
    if when:
        st.caption(f"Occurred at **{when}** UTC")
    fm = anom.first_mover()
    if fm is not None:
        onset = fm["onset"] + t_axis[0]
        onset_utc = utc_for(stamps, onset)
        st.caption(f"First to deviate: **{pretty(fm['channel'])}** "
                   f"({pretty(fm['subsystem'])}) at "
                   + (f"{onset_utc} UTC" if onset_utc else f"t={onset:,}"))

    # --- causal ordering: which channel moved first -------------------------
    chain = anom.chain()
    lead = chain[0] if chain else None
    if lead is not None:
        st.markdown("**Sequence: which channel moved first**")
        if len(chain) > 1:
            if chain[0]["lag"] == chain[1]["lag"]:
                st.info(f"{pretty(chain[0]['channel'])} and {len(chain) - 1} other channel(s) deviated simultaneously - no lead/lag separation.")
            else:
                st.info(f"{pretty(chain[0]['channel'])} led the deviation, followed by {len(chain) - 1} other channel(s).")
        else:
            st.info(f"{pretty(chain[0]['channel'])} was the only channel to deviate.")
        if len(chain) > 1:
            biggest = anom.contributions[0]["channel"]
            if lead["channel"] != biggest:
                st.warning(
                    f"The loudest channel is **{pretty(biggest)}**, but **{pretty(lead['channel'])}** "
                    "deviated first — the largest symptom is not the origin here."
                )
            rows = []
            for i, c in enumerate(chain):
                rows.append({
                    "Order": i + 1,
                    "Channel": pretty(c["channel"]),
                    "Subsystem": (MEANING.get(c["channel"], {}).get("subsystem")
                                  or c["subsystem"]),
                    "Started at": c["onset"] + t_axis[0],
                    "Lag": "first" if c["lag"] == lead["lag"]
                           else f"+{c['lag'] - lead['lag']} steps",
                    "Severity (x error)": round(c["z"], 1),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
            st.caption("Ordered by onset, not by size. Temporal precedence is "
                       "evidence that a channel is upstream — not proof of "
                       "physical causation.")
        st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Why: per-channel contribution**")
        top = anom.contributions[:8]
        fig = go.Figure(go.Bar(
            x=[c["share_pct"] for c in top][::-1],
            y=[pretty(c["channel"]) for c in top][::-1],
            orientation="h",
            marker_color=[severity_band(c["z"])[1] for c in top][::-1],
            text=[f"{c['share_pct']:.1f}%  ({c['z']:.1f}x error)" for c in top][::-1],
            textposition="auto",
            hovertemplate="%{y}: %{x:.1f}% of deviation<extra></extra>",
        ))
        fig.update_layout(height=max(220, 34 * len(top)),
                          margin=dict(l=0, r=0, t=6, b=0),
                          xaxis_title="share of total deviation (%)")
        st.plotly_chart(fig, use_container_width=True,
                        key=f"{key_prefix}contrib")

    with right:
        st.markdown("**Affected subsystems**")
        st.dataframe(
            pd.DataFrame([
                {"Subsystem": pretty(s["subsystem"]),
                 "Share": f"{s['share_pct']:.1f}%",
                 "Channels": ", ".join([pretty(ch) for ch in s["channels"][:4]])}
                for s in anom.subsystems[:5]
            ]),
            hide_index=True, use_container_width=True,
        )
        st.markdown("**Expected vs actual at peak**")
        st.dataframe(
            pd.DataFrame([
                {"Channel": pretty(c["channel"]),
                 "Expected": round(c["predicted"], 3),
                 "Actual": round(c["actual"], 3),
                 "Δ": round(c["deviation"], 3)}
                for c in anom.contributions[:5]
            ]),
            hide_index=True, use_container_width=True,
        )

    # Overlay the offending channels around the event.
    lead = [c["channel"] for c in anom.contributions[:3]]
    lo = max(0, anom.start - t_axis[0] - 250)
    hi = min(n_steps, anom.end - t_axis[0] + 250)
    fig = go.Figure()
    for ch in lead:
        j = channels.index(ch)
        fig.add_trace(go.Scatter(x=t_axis[lo:hi], y=result["y_pred"][lo:hi, j],
                                 name=f"{pretty(ch)} (Expected)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=t_axis[lo:hi], y=result["y_true"][lo:hi, j],
                                 name=f"{pretty(ch)} (Actual)"))
    fig.add_vrect(x0=anom.start, x1=anom.end, fillcolor=colour, opacity=0.18,
                  line_width=0)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.2),
                      xaxis_title="timestep", yaxis_title="telemetry value")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}trace")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Early Warning (CUSUM Drift)**")
        st.caption("**What this means:** Tracks slow, gradual component wear-and-tear before a critical failure happens. If the cumulative drift crosses the Orange Watch Threshold (5.0), it means the component is steadily failing.")
        if lead:
            top_channel = lead[0]
            j_top = channels.index(top_channel)
            err_col = result["err"][:, j_top]
            
            med = np.nanmedian(err_col)
            mad = max(np.nanmedian(np.abs(err_col - med)) * 1.4826, 1e-9)
            z_err = np.clip((err_col - med) / mad, -3.0, 3.0)
            z_err = np.nan_to_num(z_err, nan=0.0)
            s_acc = np.zeros_like(z_err)
            acc = 0.0
            for i_z, v in enumerate(z_err):
                acc = max(0.0, acc + v - 0.5)
                s_acc[i_z] = acc
                
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=t_axis[lo:hi], y=s_acc[lo:hi],
                                      name="drift", fill="tozeroy"))
            fig2.add_hline(y=5.0, line_dash="dash", line_color="#ff7f0e", annotation_text="Watch Threshold (5.0)")
            fig2.add_vrect(x0=anom.start, x1=anom.end, fillcolor=colour, opacity=0.15, line_width=0)
            fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="timestep", yaxis_title="cumulative drift")
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}cusum")

            # Predict Exact Time of Malfunction (TTF Extrapolation)
            try:
                import json
                with open("prototype/reports/early_warning.json", "r") as f:
                    ew_data = json.load(f)
                lead_time = ew_data.get(spacecraft, {}).get("cusum", {}).get("median_lead", 400)
            except:
                lead_time = 400
                
            time_str = f"**~{lead_time} timesteps**"
            dt_seconds = 60 # Default to 1 minute per timestep if no timestamps exist
            if stamps is not None and len(stamps) > 1:
                try:
                    s = pd.to_datetime(stamps)
                    diffs = s.to_series().diff().dt.total_seconds()
                    # Median delta ignores massive time gaps (e.g. satellite signal loss)
                    dt_seconds = diffs.median()
                except Exception:
                    pass
            
            if dt_seconds > 0:
                seconds = int(lead_time * dt_seconds)
                days = seconds // 86400
                hours = (seconds % 86400) // 3600
                minutes = (seconds % 3600) // 60
                
                parts = []
                if days > 0:
                    parts.append(f"{days} days")
                if hours > 0:
                    parts.append(f"{hours} hours")
                if minutes > 0 or (days == 0 and hours == 0):
                    parts.append(f"{minutes} minutes")
                
                time_str = f"**~{', '.join(parts)}** ({lead_time} timesteps)"
            
            st.warning(f"⏳ **Predictive Maintenance (Time-To-Failure):** Based on the CUSUM degradation rate, the AI predicts this sensor will suffer a critical failure in {time_str}. This provides actionable lead time to route power away from the subsystem.")


    with c_right:
        st.markdown("**Subsystem Correlation Matrix**")
        st.caption("**What this means:** Shows which sensors are failing together. Dark Red (close to 1.0) means these components are heavily linked and likely breaking due to the same root cause.")
        if len(lead) > 1:
            corr_data = {}
            for ch in lead:
                ch_idx = channels.index(ch)
                corr_data[ch] = result["y_true"][lo:hi, ch_idx]
            df_corr = pd.DataFrame(corr_data).corr()
            fig_corr = go.Figure(go.Heatmap(
                z=df_corr.values,
                x=[channel_label(c) for c in df_corr.columns],
                y=[channel_label(c) for c in df_corr.columns],
                colorscale="RdBu", zmin=-1, zmax=1,
                text=np.round(df_corr.values, 2),
                texttemplate="%{text}",
            ))
            fig_corr.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_corr, use_container_width=True, key=f"{key_prefix}corr")
        else:
            st.info("Not enough contributing channels to correlate.")


# ------------------------------------------------------------ live replay ---
if mode == "Live replay":
    st.subheader("Live replay")
    st.caption("Telemetry is streamed back at accelerated speed. Alerts fire "
               "the moment the model's forecast diverges from reality.")

    # Playback is driven by session state and an auto-rerunning fragment rather
    # than a blocking loop.  A `for ... time.sleep()` loop holds the script
    # thread for the whole run, so Streamlit cannot process a button press until
    # it finishes - which means no pause, no scrubbing and no way to stop.
    sig = (spacecraft, z_min, min_run)
    if st.session_state.get("replay_sig") != sig:
        st.session_state.replay_sig = sig
        st.session_state.replay_pos = 0
        st.session_state.replay_playing = False

    playing = st.session_state.replay_playing
    pos = st.session_state.replay_pos

    c_play, c_step, c_reset, c_speed, c_tail = st.columns([1, 1, 1, 2, 2])
    if c_play.button("⏸ Pause" if playing else "▶ Play", type="primary",
                     use_container_width=True):
        st.session_state.replay_playing = not playing
        if st.session_state.replay_playing and pos >= n_steps:
            st.session_state.replay_pos = 0
        st.rerun()
    if c_step.button("⏭ Step", use_container_width=True,
                     help="advance one frame while paused"):
        st.session_state.replay_playing = False
        st.session_state.replay_pos = min(n_steps, pos + 1)
        st.rerun()
    if c_reset.button("↺ Restart", use_container_width=True):
        st.session_state.replay_playing = False
        st.session_state.replay_pos = 0
        st.rerun()
    speed = c_speed.select_slider("Speed", [10, 25, 50, 100, 200], value=50,
                                  help="timesteps advanced per frame")
    tail = c_tail.number_input("Visible window", 200, 3000, 800, step=100)

    if anomalies:
        j1, j2 = st.columns([3, 1])
        pick = j1.selectbox(
            "Jump to incident", range(len(anomalies)),
            format_func=lambda i: (
                f"#{i + 1} · {severity_band(anomalies[i].severity)[0]} · "
                f"t={anomalies[i].start:,} · {pretty(anomalies[i].top_channel())}"),
            key="replay_jump",
        )
        if j2.button("Go to it", use_container_width=True):
            # Land a little before the incident so it is visible arriving.
            target = anomalies[pick].start - t_axis[0] + 40
            st.session_state.replay_pos = int(min(n_steps, max(1, target)))
            st.session_state.replay_playing = False
            st.rerun()

    def _y_top(visible) -> float:
        """Upper y-limit from the visible slice.

        A single channel can spike past 600 sigma, which would flatten every
        normal reading against the axis if the range were taken from the whole
        run.  Scale to what is on screen, keeping the threshold line in view.
        """
        peak = float(visible.max()) if len(visible) else 0.0
        floor = (z_min * 1.5) if z_min is not None else 4.0
        return max(floor, peak * 1.15)

    def draw(upto: int) -> None:
        upto = max(1, min(n_steps, upto))
        lo = max(0, upto - tail)
        visible = result["fleet"][lo:upto]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_axis[lo:upto], y=visible,
            name="fleet anomaly score", line=dict(color="#1f77b4", width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
        ))
        if z_min is not None:
            fig.add_hline(y=z_min, line_dash="dash", line_color="#d62728",
                          annotation_text=f"alert threshold {z_min}σ")
        x_lo, x_hi = int(t_axis[lo]), int(t_axis[upto - 1])
        for a in anomalies:
            if a.end < x_lo or a.start > x_hi:
                continue
            fig.add_vrect(x0=max(a.start, x_lo), x1=min(a.end, x_hi),
                          fillcolor=severity_band(a.severity)[1],
                          opacity=0.2, line_width=0)
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="timestep",
                          yaxis_title="anomaly severity (x normal error)",
                          yaxis_range=[0, _y_top(visible)])
        # Drawn straight into the fragment rather than into an st.empty()
        # placeholder: a placeholder created fresh on each rerun was being
        # cleared before the chart landed in it, leaving a blank gap where the
        # plot should be. The fragment already replaces its own output.
        st.plotly_chart(fig, use_container_width=True,
                        key=f"replay_chart_{upto}")

    def replay_frame() -> None:
        if st.session_state.replay_playing:
            st.session_state.replay_pos = min(n_steps,
                                              st.session_state.replay_pos + speed)
            if st.session_state.replay_pos >= n_steps:
                st.session_state.replay_playing = False

        upto = st.session_state.replay_pos
        now = int(t_axis[min(upto, n_steps) - 1])
        st.progress(upto / n_steps,
                    text=f"t = {now:,} of {int(t_axis[-1]):,}  "
                         f"({upto:,} / {n_steps:,} readings)")
        draw(upto)

        fired = [a for a in anomalies if a.start <= now]
        if fired:
            latest = max(fired, key=lambda a: a.start)
            band, _ = severity_band(latest.severity)
            if latest.contributions:
                parts = [f"{pretty(c['channel'])} ({c['share_pct']:.0f}% contribution)" for c in latest.contributions[:3]]
                lead_sys = pretty(latest.subsystems[0]["subsystem"]) if latest.subsystems else "unknown"
                exp = f"Driven by **{lead_sys}**: {', '.join(parts)}."
            else:
                exp = "No channel attribution available."
            st.error(f"**{band} ALERT** at t={latest.start:,} — {exp}")
            st.caption(f"{len(fired)} of {len(anomalies)} incidents raised so far."
                       + ("  Playback finished." if upto >= n_steps else ""))
        else:
            st.success("No anomalies raised yet — telemetry nominal.")

        if st.session_state.replay_playing:
            import time
            time.sleep(0.6)
            st.rerun()

    replay_frame()

# ---------------------------------------------------------- full timeline ---
elif mode == "Full timeline":
    st.subheader("System Tension (Early Warning)")
    if "rmt_tension" in result:
        fig_rmt = go.Figure()
        fig_rmt.add_trace(go.Scatter(
            x=t_axis, y=result["rmt_tension"], name="Tension",
            line=dict(color="#8c564b", width=1.5),
            hovertemplate="t=%{x}<br>Tension: %{y:.2f}<extra></extra>"
        ))
        fig_rmt.add_hline(y=2.0, line_dash="dash", line_color="#d62728",
                          annotation_text="Critical Tension (2.0)")
        for ch, seqs in bundle.labels.items():
            for s, e in seqs:
                fig_rmt.add_vrect(x0=s, x1=e, line_width=1, line_color="#444",
                                  fillcolor="rgba(0,0,0,0)")
        fig_rmt.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="", yaxis_title="System Tension (z)")
        st.plotly_chart(fig_rmt, use_container_width=True)
        st.caption(
            "**What this means:** Measures the overall 'stress' of the entire satellite. "
            "When the Tension line spikes above the red threshold (2.0), the satellite's systems "
            "are acting chaotically. This often acts as a mathematical 'smoke alarm', giving you an early warning days before a critical component failure."
        )
        st.divider()

    st.subheader("Fleet anomaly score")
    # Scores span three orders of magnitude - routine noise sits near 1 sigma
    # while a hard failure can reach 600.  On a linear axis the spike flattens
    # everything else onto the baseline, so plot it logarithmically.
    fleet_plot = np.maximum(result["fleet"], 0.1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_axis, y=fleet_plot, name="anomaly severity",
                             line=dict(color="#1f77b4", width=1),
                             hovertemplate="t=%{x}<br>%{y:.1f}x error<extra></extra>"))
    if z_min is not None:
        fig.add_hline(y=z_min, line_dash="dash", line_color="#d62728")
    for a in anomalies:
        fig.add_vrect(x0=a.start, x1=a.end,
                      fillcolor=severity_band(a.severity)[1],
                      opacity=0.22, line_width=0)
    for ch, seqs in bundle.labels.items():
        for s, e in seqs:
            fig.add_vrect(x0=s, x1=e, line_width=1, line_color="#444",
                          fillcolor="rgba(0,0,0,0)")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="timestep",
                      yaxis_title="anomaly severity (x normal error, log scale)",
                      yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Log scale — a single channel can exceed 600σ, which would flatten "
               "everything else on a linear axis. Shaded bands = detections "
               "(coloured by severity). Outlined bands = NASA ground-truth "
               "anomaly windows.")

    st.subheader("Per-channel deviation heatmap")
    step = max(1, n_steps // 900)
    z = result["z"][::step].T
    fig = go.Figure(go.Heatmap(
        z=z, x=t_axis[::step], y=[pretty(c) for c in channels], colorscale="Inferno",
        # The colour ceiling is FIXED at the CRITICAL band rather than scaled to
        # whatever is present. Auto-scaling made a perfectly healthy satellite
        # look alarming: on COSMO the brightest colour meant 2.4 sigma, which is
        # ordinary noise, while the same colour on SMAP meant a real 600-sigma
        # fault. Brightness now means the same thing on every satellite, so a
        # quiet spacecraft reads as quiet.
        zmin=0, zmax=CRITICAL_SIGMA,
        colorbar=dict(title="x error"),
    ))
    fig.update_layout(height=max(420, 15 * len(channels)),
                      margin=dict(l=0, r=0, t=10, b=0), xaxis_title="timestep")
    st.plotly_chart(fig, use_container_width=True)
    peak = float(np.nanmax(result["z"])) if np.isfinite(result["z"]).any() else 0.0
    st.caption(
        f"Each row is one telemetry channel. The colour scale is fixed at "
        f"0–{CRITICAL_SIGMA:.0f}x error so brightness means the same thing on every "
        f"satellite. This capture peaks at **{peak:.1f}x error** — "
        + ("mostly dark means nothing here is anomalous, which is what a healthy "
           "spacecraft should look like."
           if peak < CRITICAL_SIGMA else
           "bright bands are the channels that drove each detected event.")
    )

# ------------------------------------------------------------- anomaly log ---
else:
    st.subheader("Ranked anomaly log")
    if not anomalies:
        st.info("No anomalies at the current threshold. Switch to Manual σ "
                "and lower it in the sidebar.")
    else:
        st.dataframe(
            pd.DataFrame([{
                "#": i + 1,
                "Severity": severity_band(a.severity)[0],
                "Score (x error)": round(a.severity, 2),
                **({"When (UTC)": utc_for(stamps, a.start)} if stamps is not None
                   else {}),
                "Start": a.start, "End": a.end, "Steps": a.duration,
                "Largest": pretty(a.top_channel()),
                "Moved first": pretty((a.first_mover() or {}).get("channel", "-")),
                "Subsystem": pretty(a.subsystems[0]["subsystem"]) if a.subsystems else "-",
            } for i, a in enumerate(anomalies)]),
            hide_index=True, use_container_width=True, height=280,
        )
        pick = st.selectbox(
            "Inspect anomaly", range(len(anomalies)),
            format_func=lambda i: (
                f"#{i + 1} · {severity_band(anomalies[i].severity)[0]} · "
                f"t={anomalies[i].start:,} · {pretty(anomalies[i].top_channel())}"
            ),
        )
        st.divider()
        explanation_panel(anomalies[pick], key_prefix=f"log{pick}")
