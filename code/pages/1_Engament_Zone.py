import streamlit as st
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
sys.path.append('code/utils')
from utils.engament_zone_model import *
from utils.engament_zone_viz import *

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import io
##---Setup---##
color = st.get_option("theme.primaryColor")
os.makedirs('../data', exist_ok=True)

# Page config
st.set_page_config(page_title="Engagement Zone — Strategic Setup", layout="wide", initial_sidebar_state="expanded", page_icon="🛩️")

# Header
st.title("🧭 Page 1 — Engagement Zone (Strategic Setup)")
st.markdown(
    """
Use this page to **select scenario coordinates** and **configure Engagement Zone (EZ) parameters** before running the path planner.

- Define **Agent** state (position + heading)
- Define **Intruder** state (position) and relative speed **μ**
- Tune **EZ ranges** (**R_ez**, **r**) and review **safety/feasibility**

Your selections are stored in `st.session_state` and will be used by **Page 2 — Path Planner**.
    """
)

st.divider()

##---Sidebar---##

with st.sidebar:
    st.header("Scenario Inputs")
    st.caption("Set the strategic configuration here. Values persist across pages via session state.")

    with st.expander("Agent", expanded=True):
        col1, col2 = st.columns(2)
        agent_x = col1.number_input("X", value=float(st.session_state.get("agent_x", 2.0)), step=0.5, format="%.1f", key="agent_x")
        agent_y = col2.number_input("Y", value=float(st.session_state.get("agent_y", -1.0)), step=0.5, format="%.1f", key="agent_y")
        agent_heading = st.selectbox(
            "Heading (deg)",
            options=[-180, -135, -90, -60, -45, -30, 0, 30, 45, 60, 90, 135, 180],
            index=[-180, -135, -90, -60, -45, -30, 0, 30, 45, 60, 90, 135, 180].index(int(st.session_state.get("agent_heading", -30))),
            key="agent_heading",
            format_func=lambda x: f"{x}°",
        )
        st.session_state.agent_parameters = agent(agent_x, agent_y, agent_heading)

    with st.expander("Intruder", expanded=True):
        col1, col2 = st.columns(2)
        intruder_x = col1.number_input("X", value=float(st.session_state.get("intruder_x", 0.0)), step=0.5, format="%.1f", key="intruder_x")
        intruder_y = col2.number_input("Y", value=float(st.session_state.get("intruder_y", 0.0)), step=0.5, format="%.1f", key="intruder_y")
        mu = st.slider("Speed Rate (μ)", min_value=0.1, max_value=1.0, value=float(st.session_state.get("mu", 0.5)), step=0.1, key="mu")

    with st.expander("Engagement Zone", expanded=True):
        R_ez = st.slider("Maximum Range (R_ez)", min_value=1.0, max_value=10.0, value=float(st.session_state.get("R_ez", 5.0)), step=1.0, key="R_ez")
        r = st.slider("Minimum Range (r)", min_value=0.0, max_value=5.0, value=float(st.session_state.get("r", 0.0)), step=0.25, key="r")

    st.divider()
    st.caption("Tip: After validating safety/feasibility here, go to **Page 2 — Path Planner** to generate an EZ-aware path.")

# Store intruder parameters in session state
st.session_state.intruder_parameters = intruder(x=st.session_state.get("intruder_x", 0.0), y=st.session_state.get("intruder_y", 0.0))

# Store EZ data in session state
st.session_state.xi_values, st.session_state.ez_points, st.session_state.ez_df = rho_values(mu, R_ez, r, save_csv='../data/ez_boundary.csv', save_txt='../data/ez_boundary.txt')

# Always create boundaries figure for display and download
safe, message, summary_df = agent_safety(st.session_state.intruder_parameters, st.session_state.agent_parameters, mu, R_ez, r)
ez_parameters = dict(zip(summary_df["Parameter"], summary_df["Value"]))
ez_parameters['safe'] = safe
ez_parameters['message'] = message
st.session_state.ez_parameters = ez_parameters
st.session_state.safe = safe
st.session_state.message = message

# --- Status summary -----------------------------------------------------------
status_cols = st.columns([1.2, 1.2, 1.2, 2.4])
with status_cols[0]:
    st.metric("μ", f"{float(mu):.2f}")
with status_cols[1]:
    st.metric("R_ez", f"{float(R_ez):.2f}")
with status_cols[2]:
    st.metric("r", f"{float(r):.2f}")
with status_cols[3]:
    if safe:
        st.success(f"SAFE: {message}")
    else:
        st.error(f"UNSAFE: {message}")

st.divider()

fig_boundaries = plot_engagement_zone_boundaries(mu, R_ez, r, st.session_state.intruder_parameters, st.session_state.xi_values, st.session_state.ez_points, figsize=(14, 12))
fig_interaction = plot_engagement_zone_interaction(ez_parameters, (st.session_state.xi_values, st.session_state.ez_points, st.session_state.ez_df), figsize=(14, 12))


left, right = st.columns([2.6, 1.4], gap="large")

with left:
    tab_b, tab_i, tab_s = st.tabs(["📉 Boundaries", "🎯 Interaction", "🧾 Summary"])

    with tab_b:
        st.pyplot(fig_boundaries, use_container_width=True)

    with tab_i:
        st.pyplot(fig_interaction, use_container_width=True)

    with tab_s:
        st.subheader("EZ Summary Table")
        st.dataframe(st.session_state.ez_df, use_container_width=True)

with right:
    st.subheader("Downloads")
    st.caption("Export figures for reports or sharing.")

    buf_boundaries = io.BytesIO()
    fig_boundaries.savefig(buf_boundaries, format="png")
    buf_boundaries.seek(0)
    st.download_button(
        label="Download Boundaries (PNG)",
        data=buf_boundaries,
        file_name="engagement_zone_boundaries.png",
        mime="image/png",
        icon="📉",
    )

    buf_interaction = io.BytesIO()
    fig_interaction.savefig(buf_interaction, format="png")
    buf_interaction.seek(0)
    st.download_button(
        label="Download Interaction (PNG)",
        data=buf_interaction,
        file_name="engagement_zone_interaction.png",
        mime="image/png",
        icon="🎯",
    )

    st.divider()
    st.subheader("Derived Variables")
    parameters = agent_safety(
        st.session_state.intruder_parameters,
        st.session_state.agent_parameters,
        st.session_state.mu,
        st.session_state.R_ez,
        st.session_state.r,
    )
    st.dataframe(parameters[2], use_container_width=True)
