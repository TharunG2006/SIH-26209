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

from config import (MIN_RUN, SPACECRAFT, Z_MIN, channel_label,  # noqa: E402
                    has_operator_names)
from data import load_spacecraft  # noqa: E402
from detect import detect  # noqa: E402

st.set_page_config(page_title="Satellite Health Monitor", layout="wide",
                   page_icon="🛰️")

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
    bundle = load_spacecraft(spacecraft)
    result = detect(bundle, z_min=z_min, min_run=min_run)
    return bundle, result


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("🛰️ Mission Control")
spacecraft = st.sidebar.selectbox(
    "Spacecraft", list(SPACECRAFT),
    format_func=lambda s: SPACECRAFT[s]["label"],
)
threshold_mode = st.sidebar.radio(
    "Threshold", ["Automatic (recommended)", "Manual σ"],
    help="Automatic gives each channel its own nonparametric threshold and "
         "prunes weak sequences — this is the mode the reported metrics use.",
)
if threshold_mode.startswith("Automatic"):
    z_min = None
    st.sidebar.caption("Each channel's threshold is chosen from its own error "
                       "distribution, so a noisy channel cannot drown out a "
                       "quiet one.")
else:
    z_min = st.sidebar.slider("Detection threshold (σ)", 1.5, 8.0, float(Z_MIN),
                              0.1, help="Higher = fewer, more confident alerts")
min_run = st.sidebar.slider("Min. duration (timesteps)", 1, 20, MIN_RUN)

bundle, result = load(spacecraft, z_min, min_run)
channels = result["channels"]
t_axis = result["t"]
n_steps = len(t_axis)

st.sidebar.divider()
mode = st.sidebar.radio("View", ["Live replay", "Full timeline", "Anomaly log"])

# ------------------------------------------------------------------ header ---
st.title("Explainable Satellite Anomaly Detection")
st.caption(
    f"{SPACECRAFT[spacecraft]['label']} — {len(channels)} telemetry channels, "
    f"{n_steps:,} scored timesteps, forecast by the {result['forecaster']}. "
    "Anomalies are attributed back to the channels that caused them, and "
    "ordered by which channel deviated first."
)

if not has_operator_names():
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
c5.metric("Ground-truth windows",
          sum(len(v) for v in bundle.labels.values()))


def explanation_panel(anom, key_prefix: str = "") -> None:
    """Render the explainability card for one anomaly."""
    band, colour = severity_band(anom.severity)
    st.markdown(
        f"<div style='border-left:6px solid {colour};padding:0.6rem 1rem;"
        f"background:rgba(128,128,128,0.08);border-radius:4px'>"
        f"<b style='color:{colour}'>{band}</b> &nbsp; "
        f"t = {anom.start:,} → {anom.end:,} ({anom.duration} steps) &nbsp;·&nbsp; "
        f"peak {anom.severity:.1f}σ at t={anom.peak:,}<br>"
        f"<span style='font-size:0.95em'>{anom.explanation()}</span></div>",
        unsafe_allow_html=True,
    )
    fm = anom.first_mover()
    if fm is not None:
        st.caption(f"First to deviate: **{fm.get('label', fm['channel'])}** "
                   f"({fm['subsystem']}) at t={fm['onset'] + t_axis[0]:,}")

    # --- causal ordering: which channel moved first -------------------------
    chain = anom.chain()
    lead = chain[0] if chain else None
    if lead is not None:
        st.markdown("**Sequence: which channel moved first**")
        st.info(anom.propagation())
        if len(chain) > 1:
            biggest = anom.contributions[0]["channel"]
            if lead["channel"] != biggest:
                st.warning(
                    f"The loudest channel is **{biggest}**, but **{lead['channel']}** "
                    "deviated first — the largest symptom is not the origin here."
                )
            rows = []
            for i, c in enumerate(chain):
                rows.append({
                    "Order": i + 1,
                    "Channel": c.get("label", c["channel"]),
                    "Subsystem": c["subsystem"],
                    "Started at": c["onset"] + t_axis[0],
                    "Lag": "first" if c["lag"] == lead["lag"]
                           else f"+{c['lag'] - lead['lag']} steps",
                    "Severity (σ)": round(c["z"], 1),
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
            y=[c.get("label", c["channel"]) for c in top][::-1],
            orientation="h",
            marker_color=[severity_band(c["z"])[1] for c in top][::-1],
            text=[f"{c['share_pct']:.1f}%  ({c['z']:.1f}σ)" for c in top][::-1],
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
                {"Subsystem": s["subsystem"],
                 "Share": f"{s['share_pct']:.1f}%",
                 "Channels": ", ".join(s["channels"][:4])}
                for s in anom.subsystems[:5]
            ]),
            hide_index=True, use_container_width=True,
        )
        st.markdown("**Expected vs actual at peak**")
        st.dataframe(
            pd.DataFrame([
                {"Channel": c.get("label", c["channel"]),
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
                                 name=f"{ch} expected", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=t_axis[lo:hi], y=result["y_true"][lo:hi, j],
                                 name=f"{ch} actual"))
    fig.add_vrect(x0=anom.start, x1=anom.end, fillcolor=colour, opacity=0.18,
                  line_width=0)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.2),
                      xaxis_title="timestep", yaxis_title="telemetry value")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}trace")


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
        st.session_state.replay_pos = min(n_steps, 800)
        st.session_state.replay_playing = False

    playing = st.session_state.replay_playing
    pos = st.session_state.replay_pos

    c_play, c_step, c_reset, c_speed, c_tail = st.columns([1, 1, 1, 2, 2])
    if c_play.button("⏸ Pause" if playing else "▶ Play", type="primary",
                     use_container_width=True):
        st.session_state.replay_playing = not playing
        st.rerun()
    if c_step.button("⏭ Step", use_container_width=True,
                     help="advance one frame while paused"):
        st.session_state.replay_playing = False
        st.session_state.replay_pos = min(n_steps, pos + 1)
        st.rerun()
    if c_reset.button("↺ Restart", use_container_width=True):
        st.session_state.replay_playing = False
        st.session_state.replay_pos = min(n_steps, 800)
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
                f"t={anomalies[i].start:,} · {anomalies[i].top_channel()}"),
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

    def draw(upto: int, slot) -> None:
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
                          yaxis_title="anomaly score (σ)",
                          yaxis_range=[0, _y_top(visible)])
        slot.plotly_chart(fig, use_container_width=True, key="replay_chart")

    # run_every is re-evaluated on every script run, so toggling the play flag
    # starts and stops the auto-advance without restarting the app.
    @st.fragment(run_every="0.25s" if st.session_state.replay_playing else None)
    def replay_frame() -> None:
        if st.session_state.replay_playing:
            st.session_state.replay_pos = min(n_steps,
                                              st.session_state.replay_pos + speed)
            if st.session_state.replay_pos >= n_steps:
                st.session_state.replay_playing = False
                st.rerun()          # refresh the Play/Pause label

        upto = st.session_state.replay_pos
        now = int(t_axis[min(upto, n_steps) - 1])
        st.progress(upto / n_steps,
                    text=f"t = {now:,} of {int(t_axis[-1]):,}  "
                         f"({upto:,} / {n_steps:,} readings)")
        draw(upto, st.empty())

        fired = [a for a in anomalies if a.start <= now]
        if fired:
            latest = max(fired, key=lambda a: a.start)
            band, _ = severity_band(latest.severity)
            st.error(f"**{band} ALERT** at t={latest.start:,} — "
                     f"{latest.explanation()}")
            st.caption(f"{len(fired)} of {len(anomalies)} incidents raised so far."
                       + ("  Playback finished." if upto >= n_steps else ""))
        else:
            st.success("No anomalies raised yet — telemetry nominal.")

    replay_frame()

# ---------------------------------------------------------- full timeline ---
elif mode == "Full timeline":
    st.subheader("Fleet anomaly score")
    # Scores span three orders of magnitude - routine noise sits near 1 sigma
    # while a hard failure can reach 600.  On a linear axis the spike flattens
    # everything else onto the baseline, so plot it logarithmically.
    fleet_plot = np.maximum(result["fleet"], 0.1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_axis, y=fleet_plot, name="anomaly score",
                             line=dict(color="#1f77b4", width=1),
                             hovertemplate="t=%{x}<br>%{y:.1f}σ<extra></extra>"))
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
                      yaxis_title="anomaly score (σ, log scale)",
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
        z=z, x=t_axis[::step], y=channels, colorscale="Inferno",
        # Shorter channels are NaN-padded, and np.percentile propagates NaN
        # straight into the colour scale.
        zmin=0, zmax=float(np.nanpercentile(result["z"], 99.7)),
        colorbar=dict(title="σ"),
    ))
    fig.update_layout(height=max(420, 15 * len(channels)),
                      margin=dict(l=0, r=0, t=10, b=0), xaxis_title="timestep")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each row is one telemetry channel. Bright bands show exactly "
               "which channels drove each event — this is the attribution the "
               "alert cards are built from.")

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
                "Score (σ)": round(a.severity, 2),
                "Start": a.start, "End": a.end, "Steps": a.duration,
                "Largest": a.top_channel(),
                "Moved first": (a.first_mover() or {}).get("channel", "-"),
                "Subsystem": a.subsystems[0]["subsystem"] if a.subsystems else "-",
            } for i, a in enumerate(anomalies)]),
            hide_index=True, use_container_width=True, height=280,
        )
        pick = st.selectbox(
            "Inspect anomaly", range(len(anomalies)),
            format_func=lambda i: (
                f"#{i + 1} · {severity_band(anomalies[i].severity)[0]} · "
                f"t={anomalies[i].start:,} · {anomalies[i].top_channel()}"
            ),
        )
        st.divider()
        explanation_panel(anomalies[pick], key_prefix=f"log{pick}")
