import streamlit as st
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import math
import time
from pathlib import Path

sys.path.append('code/utils')
from utils.engament_zone_model import *
from utils.engament_zone_viz import *
from utils.rrt_path_planner import RRTStar

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Page configuration
st.set_page_config(page_title="Path Planner with EZ Avoidance", layout="wide", page_icon="🛩️")

##---Helper Functions---##

def fit_circle(x_coords, y_coords):
    """Compute the center (a, b) and radius of a circle passing through the given points."""
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    if x.size < 3:
        raise ValueError("Need at least three points to determine EZ circle.")
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    rhs = x ** 2 + y ** 2
    solution, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = solution
    radius = math.sqrt(a ** 2 + b ** 2 + c)
    return float(a), float(b), float(radius)


def compute_workspace(start, goal, obstacles, padding=0.15):
    """Shift start/goal/obstacle coordinates to fit inside the planner's [0, map_size] box."""
    xs = [start[0], goal[0]]
    ys = [start[1], goal[1]]
    for (ox, oy, rad) in obstacles:
        xs.extend([ox - rad, ox + rad])
        ys.extend([oy - rad, oy + rad])

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1e-3)
    map_span = span * (1.0 + padding)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    shift_x = (map_span / 2.0) - cx
    shift_y = (map_span / 2.0) - cy

    shifted_start = [start[0] + shift_x, start[1] + shift_y]
    shifted_goal = [goal[0] + shift_x, goal[1] + shift_y]
    shifted_obstacles = [(ox + shift_x, oy + shift_y, rad) for (ox, oy, rad) in obstacles]
    map_size = [map_span, map_span]

    return shifted_start, shifted_goal, shifted_obstacles, map_size, shift_x, shift_y


def _lookup_rho_for_angle(angle_rad, xi_values, ez_points):
    """Lookup rho value for a given angle from precomputed EZ boundary."""
    xi_wrapped = ((angle_rad + np.pi) % (2.0 * np.pi)) - np.pi
    xi_arr = np.asarray(xi_values, dtype=float)
    rho_arr = np.asarray(ez_points, dtype=float)
    idx = int(np.argmin(np.abs(xi_arr - xi_wrapped)))
    return float(rho_arr[idx])


def compute_path_diagnostics(path_points, intruder_pos, xi_values, ez_points):
    """Compute diagnostic metrics for the planned path."""
    pts = np.asarray(path_points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None

    n = pts.shape[0]
    t = np.arange(n, dtype=float)
    headings = np.zeros(n)
    if n >= 2:
        deltas = np.diff(pts, axis=0)
        seg_headings = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
        headings[:-1] = seg_headings
        headings[-1] = seg_headings[-1]
    else:
        headings[:] = 0.0

    intr_x, intr_y = intruder_pos
    dx = pts[:, 0] - intr_x
    dy = pts[:, 1] - intr_y
    los_deg = np.degrees(np.arctan2(dy, dx))

    aspect_deg = np.array([wrap_angle_deg(h - l) for h, l in zip(headings, los_deg)], dtype=float)
    aspect_rad = np.deg2rad(aspect_deg)

    rho_vals = np.array([_lookup_rho_for_angle(angle, xi_values, ez_points) for angle in aspect_rad])
    distances = np.hypot(dx, dy)
    clearances = distances - rho_vals
    inside_flags = distances <= rho_vals + 1e-9

    D_max = float(np.max(clearances)) if clearances.size else 0.0
    if D_max > 0:
        C_safety = clearances / D_max
    else:
        C_safety = np.zeros_like(clearances)

    import pandas as pd
    diagnostics = pd.DataFrame({
        't': t,
        'x': pts[:, 0],
        'y': pts[:, 1],
        'heading_deg': headings,
        'LOS_deg': los_deg,
        'aspect_deg': aspect_deg,
        'distance_LOS': distances,
        'rho_xi': rho_vals,
        'clearance': clearances,
        'inside_EZ': inside_flags,
        'C_safety': C_safety,
    })
    return diagnostics


def summarize_path_metrics(diagnostics_df):
    """Summarize key metrics from the path diagnostics."""
    if diagnostics_df is None or diagnostics_df.empty:
        return {}
    clearances = diagnostics_df['clearance'].values
    inside = diagnostics_df['inside_EZ'].values
    xs = diagnostics_df['x'].values
    ys = diagnostics_df['y'].values
    if xs.size >= 2:
        step_dists = np.hypot(np.diff(xs), np.diff(ys))
        path_length = float(np.sum(step_dists))
    else:
        path_length = 0.0
    safe_points = np.count_nonzero(~inside)
    total_points = inside.size
    summary = {
        'path_length': path_length,
        'mean_clearance': float(np.mean(clearances)),
        'min_clearance': float(np.min(clearances)),
        'num_points': int(total_points),
        'num_inside_EZ': int(np.count_nonzero(inside)),
        'percent_safe_points': float(100.0 * safe_points / total_points) if total_points else 0.0,
    }
    return summary


##---Main Page---##

st.title("🛩️ Path Planner - Optimal Route Planning")

st.sidebar.header("🎯 Path Planning Configuration")

# ===== AGENT CONFIGURATION =====
st.sidebar.subheader("Agent Configuration")
agent_x = st.sidebar.number_input("Agent X Position", value=1.0, step=0.5, format="%.1f")
agent_y = st.sidebar.number_input("Agent Y Position", value=1.0, step=0.5, format="%.1f")
agent_heading = st.sidebar.slider("Agent Heading (°)", min_value=-180, max_value=180, value=45, step=15)
agent_speed = st.sidebar.number_input("Agent Speed", value=50.0, step=1.0)

# ===== INTRUDER CONFIGURATION =====
st.sidebar.subheader("Intruder Configuration")
intruder_x = st.sidebar.number_input("Intruder X Position", value=10.0, step=0.5, format="%.1f")
intruder_y = st.sidebar.number_input("Intruder Y Position", value=10.0, step=0.5, format="%.1f")
intruder_speed = st.sidebar.number_input("Intruder Speed", value=70.0, step=1.0)

# ===== ENGAGEMENT ZONE PARAMETERS =====
st.sidebar.subheader("Engagement Zone Parameters")
mu = agent_speed/intruder_speed if intruder_speed > 0 else 0.1
R_ez = st.sidebar.slider("Maximum Range (R_ez)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
r = st.sidebar.slider("Minimum Range (r)", min_value=0.0, max_value=5.0, value=0.0, step=0.25)

# ===== GOAL CONFIGURATION =====
st.sidebar.subheader("Goal Configuration")
goal_x = st.sidebar.number_input("Goal X", value=18.0, step=0.5, format="%.1f")
goal_y = st.sidebar.number_input("Goal Y", value=18.0, step=0.5, format="%.1f")

# ===== PLANNER PARAMETERS =====
st.sidebar.subheader("RRT* Planner Parameters")
col1, col2 = st.sidebar.columns(2)
step_size = col1.number_input("Step Size", value=1.0, step=1.0, format="%.1f")
search_radius = col2.number_input("Search Radius", value=4.0, step=1.0, format="%.1f")

col3, col4 = st.sidebar.columns(2)
max_iter = col3.number_input("Max Iterations", value=5000, step=100)
lambda_weight = col4.number_input("Lambda Weight", value=0.5, step=0.1, format="%.1f")

run_planner_sidebar = st.sidebar.button("🚀 Run Path Planner", type="primary", key="sidebar_button")

# Display current configuration
with st.expander("📋 Current Configuration"):
    col_agent, col_intruder, col_ez, col_goal = st.columns(4)
    
    with col_agent:
        st.subheader("Agent")
        st.write(f"Position: ({agent_x:.1f}, {agent_y:.1f})")
        st.write(f"Heading: {agent_heading:.0f}°")
        st.write(f"Speed: {agent_speed:.1f}")
    
    with col_intruder:
        st.subheader("Intruder")
        st.write(f"Position: ({intruder_x:.1f}, {intruder_y:.1f})")
        st.write(f"Speed: {intruder_speed:.1f}")
    
    with col_ez:
        st.subheader("EZ Parameters")
        st.write(f"μ = {mu:.2f}")
        st.write(f"R_ez = {R_ez:.1f}")
        st.write(f"r = {r:.2f}")
    
    with col_goal:
        st.subheader("Goal")
        st.write(f"Position: ({goal_x:.1f}, {goal_y:.1f})")

# Main button in the page
st.divider()
col_run_main, col_space = st.columns([2, 3])

# Run the planner
if run_planner_sidebar:
    # Compute EZ boundary from parameters
    xi_values, ez_points, _ = rho_values(mu, R_ez, r)
    
    if xi_values is None or ez_points is None or len(xi_values) == 0 or len(ez_points) == 0:
        st.error("❌ Cannot compute EZ boundary. Invalid parameters.")
    else:
        with st.spinner("🔄 Planning optimal path..."):
            start = [agent_x, agent_y]
            goal = [goal_x, goal_y]

            # Compute EZ circle
            ez_arr = np.asarray(ez_points, dtype=float)
            xi_arr = np.asarray(xi_values, dtype=float)
            if ez_arr.size == 0 or xi_arr.size == 0:
                st.error("Cannot compute EZ geometry - empty points")
            else:
                bx = ez_arr * np.cos(xi_arr)
                by = ez_arr * np.sin(xi_arr)
                rel_cx, rel_cy, ez_radius = fit_circle(bx, by)
                ez_center = (intruder_x + rel_cx, intruder_y + rel_cy)
                obstacles = [(ez_center[0], ez_center[1], ez_radius)]

                # Compute workspace
                shifted_start, shifted_goal, shifted_obstacles, map_size, shift_x, shift_y = compute_workspace(
                    start, goal, obstacles
                )

                intruder_shifted = {
                    'x': intruder_x + shift_x,
                    'y': intruder_y + shift_y,
                }

                # Run RRT*
                start_time = time.perf_counter()
                rrt = RRTStar(
                    shifted_start,
                    shifted_goal,
                    map_size=map_size,
                    obstacles=shifted_obstacles,
                    intruder_parameters=intruder_shifted,
                    mu=mu,
                    R=R_ez,
                    r=r,
                    agent_speed=agent_speed,
                    lambda_weight=lambda_weight,
                    step_size=step_size,
                    search_radius=search_radius,
                    max_iter=max_iter,
                    edge_sample_points=15,
                )
                
                path = rrt.plan()
                elapsed = time.perf_counter() - start_time

                st.session_state.planning_time = elapsed
                st.session_state.rrt_result = rrt
                st.session_state.path = path
                st.session_state.shift_x = shift_x
                st.session_state.shift_y = shift_y
                st.session_state.map_size = map_size
                st.session_state.shifted_obstacles = shifted_obstacles
                st.session_state.intruder_shifted = intruder_shifted
                st.session_state.xi_values = xi_values
                st.session_state.ez_points = ez_points
                st.session_state.intruder_x = intruder_x
                st.session_state.intruder_y = intruder_y

                # Show results
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = elapsed % 60
                readable = f"{hours}h {minutes}m {seconds:.2f}s"
                
                st.success(f"✅ Planning completed in {elapsed:.2f}s ({readable})")

# Display results if available
if hasattr(st.session_state, 'rrt_result'):
    st.divider()
    st.subheader("📊 Results")
    
    col_time, col_nodes, col_path = st.columns(3)
    with col_time:
        st.metric("Planning Time", f"{st.session_state.planning_time:.2f}s")
    with col_nodes:
        st.metric("Tree Nodes", len(st.session_state.rrt_result.node_list))
    with col_path:
        if st.session_state.path:
            st.metric("Path Length", f"{len(st.session_state.path):.0f} points")
        else:
            st.metric("Path Found", "❌ No")

    # Plot
    fig, _ = st.session_state.rrt_result.plot(
        intruder=(st.session_state.intruder_shifted['x'], st.session_state.intruder_shifted['y']),
        R=R_ez,
        r=r,
        origin_shift=(st.session_state.shift_x, st.session_state.shift_y),
    )
    st.pyplot(fig, use_container_width=True)

    # Path diagnostics
    if st.session_state.path:
        unshifted = [[p[0] - st.session_state.shift_x, p[1] - st.session_state.shift_y] for p in st.session_state.path]
        diagnostics = compute_path_diagnostics(
            unshifted, 
            (intruder_x, intruder_y), 
            st.session_state.xi_values, 
            st.session_state.ez_points
        )
        
        if diagnostics is not None:
            metrics = summarize_path_metrics(diagnostics)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Path Length", f"{metrics['path_length']:.2f}")
            with col_m2:
                st.metric("Mean Clearance", f"{metrics['mean_clearance']:.3f}")
            with col_m3:
                st.metric("Min Clearance", f"{metrics['min_clearance']:.3f}")
            with col_m4:
                safe_pct = metrics['percent_safe_points']
                st.metric("Safe Points", f"{safe_pct:.1f}%")
            
            # Download diagnostics
            st.download_button(
                label="📥 Download Path Diagnostics (CSV)",
                data=diagnostics.to_csv(index=False),
                file_name="path_diagnostics.csv",
                mime="text/csv"
            )

