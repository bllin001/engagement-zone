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

# Set layout to wide mode, and light theme
st.set_page_config(page_title="Engagement Zone Interactive", layout="wide", initial_sidebar_state="expanded", page_icon="🛩️")

# Set icon


##---Sidebar---##

with st.sidebar:
    show_agent = st.checkbox("Show Agent Parameters", value=True)
    show_intruder = st.checkbox("Show Intruder Parameters", value=True)

    if show_agent:
        st.header("Agent")
        col1, col2 = st.columns(2)
        agent_x = col1.number_input("X", value=2.0, step=0.5, format="%.1f", key="agent_x")
        agent_y = col2.number_input("Y", value=-1.0, step=0.5, format="%.1f", key="agent_y")
        agent_heading = st.selectbox(
            "Agent Heading (degrees)",
            options=[-180, -135, -90, -60, -45, -30, 0, 30, 45, 60, 90, 135, 180],
            index=7,  # Default to 135 degrees
            key="agent_heading",
            format_func=lambda x: f"{x}°"
        )
        agent_parameters = agent(agent_x, agent_y, agent_heading)    
    else:
        agent_x = st.session_state.get("agent_x", 2.0)
        agent_y = st.session_state.get("agent_y", -1.0)
        agent_heading = st.session_state.get("agent_heading", -30)
        agent_parameters = agent(agent_x, agent_y, agent_heading)    

    if show_intruder:
        st.header("Intruder")
        # intruder_x = st.number_input("Intruder X Position", value=0.0, step=0.5, format="%.1f", key="intruder_x")
        # intruder_y = st.number_input("Intruder Y Position", value=0.0, step=0.5, format="%.1f", key="intruder_y")
        mu = st.slider("Speed Rate ($μ$)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="mu")
        R_ez = st.slider("Maximum Range ($R_{ez}$)", min_value=1.0, max_value=10.0, value=1.0, step=1.0, format="%.1f", key="R_ez")
        r = st.slider("Minimum Range ($r$)", min_value=1.0, max_value=5.0, value=0.0, step=0.25, format="%.2f", key="r")
        
    else:
        mu = st.session_state.get("mu", 0.5)
        R_ez = st.session_state.get("R_ez", 5.0)
        r = st.session_state.get("r", 0.0)

intruder_parameters = intruder(x=0, y=0)
xi_values, ez_points, ez_df = rho_values(mu, R_ez, r, save_csv='../data/ez_boundary.csv', save_txt='../data/ez_boundary.txt')

# Always create boundaries figure for display and download
safe, message, summary_df = agent_safety(intruder_parameters, agent_parameters, mu, R_ez, r)
ez_parameters = dict(zip(summary_df["Parameter"], summary_df["Value"]))
ez_parameters['safe'] = safe
ez_parameters['message'] = message
fig_boundaries = plot_engagement_zone_boundaries(mu, R_ez, r, intruder_parameters, xi_values, ez_points, figsize=(14, 12))
fig_interaction = plot_engagement_zone_interaction(ez_parameters, (xi_values, ez_points, ez_df), figsize=(14, 12))


col_display, col_download = st.columns([3, 1])

with col_display:
    # Compute radio default index from session state without pre-setting the key
    default_selected = st.session_state.get("selected_info", "EZ Interaction")
    selected_info = st.radio(
        "Display Options",
        ["EZ Boundaries", "EZ Interaction", "EZ Summary", "EZ Variables"],
        horizontal=True,
        index=["EZ Boundaries", "EZ Interaction", "EZ Summary", "EZ Variables"].index(default_selected),
        key="selected_info"
    )

with col_download:
    if st.session_state.selected_info == "EZ Boundaries":
        buf_boundaries = io.BytesIO()
        fig_boundaries.savefig(buf_boundaries, format="png")
        buf_boundaries.seek(0)
        st.download_button(
            label="Download EZ Boundary (PNG)",
            data=buf_boundaries,
            file_name="engagement_zone_boundaries.png",
            mime="image/png",
            help="Download the engagement zone boundaries as a PNG image.",
            icon="📉"
        )
    elif st.session_state.selected_info == "EZ Interaction":
        buf_interaction = io.BytesIO()
        fig_interaction.savefig(buf_interaction, format="png")
        buf_interaction.seek(0)
        st.download_button(
            label="Download EZ Zone Interaction (PNG)",
            data=buf_interaction,
            file_name="engagement_zone_interaction.png",
            mime="image/png"
        )

if st.session_state.selected_info == "EZ Boundaries":
    with st.container():
        st.pyplot(fig_boundaries, use_container_width=False)

elif st.session_state.selected_info == "EZ Interaction":
    with st.container():
        st.pyplot(fig_interaction, use_container_width=True)

elif st.session_state.selected_info == "EZ Summary":
    with st.container():
        st.dataframe(ez_df)

elif st.session_state.selected_info == "EZ Variables":
    with st.container():
        parameters = agent_safety(intruder_parameters, agent_parameters, mu, R_ez, r)
        st.dataframe(parameters[2])
