import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
import warnings
import streamlit.components.v1 as components
from io import BytesIO
import tempfile
from matplotlib.animation import PillowWriter

from utils.engament_zone_model import agent, intruder, rho_values
from utils.rrt_path_planner import RRTStar
from utils import plot_results as pr

# All helper functions moved here (fit_circle, compute_workspace, diagnostics, etc.)
from utils.config_planner_sim import (
    fit_circle,
    compute_workspace,
    compute_path_diagnostics,
    summarize_path_metrics,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

st.set_page_config(page_title="Path Planner with EZ Avoidance", layout="wide", page_icon="🛩️")
st.title("🛩️ Path Planner - Optimal Route Planning")

st.markdown(
        """
        <style>
            /* Reduce top padding and use full width */
            .block-container { padding-top: 1rem; }

            /* Animation container: responsive, scroll when needed */
            .anim-wrap {
                width: 100%;
                max-width: 100%;
                overflow: auto;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 10px;
                padding: 0.75rem;
                background: rgba(0,0,0,0.02);
            }

            /* Allow scaling wrapper */
            .anim-scale {
                display: inline-block;
                transform-origin: 0 0;
            }

            /* Ensure JSHTML content doesn't get clipped */
            .anim-wrap canvas, .anim-wrap svg, .anim-wrap img, .anim-wrap video {
                max-width: 100% !important;
                height: auto !important;
            }

            /* Small helper text spacing */
            .anim-hint {
                margin-top: 0.25rem;
                opacity: 0.85;
            }
        </style>
        """,
        unsafe_allow_html=True,
)

# APP_ROOT points to /code directory
APP_ROOT = Path(__file__).resolve().parent.parent
# Wind data path (relative to repo root)
wind_data = (APP_ROOT.parent / "env/output/harvey_wind_probabilities_for_EZ.csv").resolve()

st.sidebar.header("🎯 Path Planning Configuration")

# ===== AGENT CONFIGURATION =====
st.sidebar.subheader("Agent")
agent_x = st.sidebar.number_input("Agent X Position", value=1.0, step=1.0, format="%.1f")
agent_y = st.sidebar.number_input("Agent Y Position", value=1.0, step=1.0, format="%.1f")
agent_heading = st.sidebar.slider("Agent Heading (°)", min_value=-180, max_value=180, value=45, step=15)
agent_speed = st.sidebar.number_input("Agent Speed", value=50.0, step=1.0)

agent_parameters = agent(x=agent_x, y=agent_y, heading=agent_heading, speed=agent_speed)

# ===== INTRUDER CONFIGURATION =====
st.sidebar.subheader("Intruder")
intruder_x = st.sidebar.number_input("Intruder X Position", value=10.0, step=1.0, format="%.1f")
intruder_y = st.sidebar.number_input("Intruder Y Position", value=10.0, step=1.0, format="%.1f")
intruder_speed = st.sidebar.number_input("Intruder Speed", value=70.0, step=1.0)

# main.py logic: intruder moves toward agent start
st.sidebar.subheader("Intruder Motion (main.py)")
move_intruder = st.sidebar.checkbox("Move intruder toward agent start", value=True)
index_speed = st.sidebar.number_input("Index Speed (units/s)", value=0.3, step=0.05, format="%.2f")

intruder_parameters = intruder(x=intruder_x, y=intruder_y, speed=intruder_speed)

# ===== ENGAGEMENT ZONE PARAMETERS =====
st.sidebar.subheader("Engagement Zone")
mu = agent_speed / intruder_speed if intruder_speed > 0 else 0.1
R_ez = st.sidebar.slider("Maximum Range (R)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
r = st.sidebar.slider("Minimum Range (r)", min_value=0.0, max_value=5.0, value=0.0, step=0.25)

# ===== GOAL CONFIGURATION =====
st.sidebar.subheader("Goal")
goal_x = st.sidebar.number_input("Goal X", value=18.0, step=0.5, format="%.1f")
goal_y = st.sidebar.number_input("Goal Y", value=18.0, step=0.5, format="%.1f")

# ===== ENV / WORKSPACE CONFIG =====
st.sidebar.subheader("Workspace (main.py option)")
workspace_mode = st.sidebar.selectbox(
    "Workspace mode",
    ["Fixed 50x50 (main.py)", "Auto-fit (old Streamlit)"],
    index=0
)
fixed_map = (workspace_mode == "Fixed 50x50 (main.py)")

# ===== WIND CONFIG =====
st.sidebar.subheader("Wind")
use_wind = st.sidebar.checkbox("Use wind uncertainty CSV", value=True)
if use_wind and not wind_data.exists():
    st.sidebar.warning("Wind CSV not found at:\n" + str(wind_data))

# ===== PLANNER PARAMETERS =====
st.sidebar.subheader("RRT* Planner")
col1, col2 = st.sidebar.columns(2)
step_size = col1.number_input("Step Size", value=1.0, step=0.5, format="%.2f")
search_radius = col2.number_input("Search Radius", value=1.0, step=0.5, format="%.2f")

col3, col4 = st.sidebar.columns(2)
max_iter = col3.number_input("Max Iterations", value=100000, step=5000)
lambda_weight = col4.number_input("Lambda Weight", value=0.5, step=0.1, format="%.2f")

edge_sample_points = st.sidebar.number_input("Edge Sample Points", value=15, step=1)

export_sim = st.sidebar.checkbox("Export simulation CSVs", value=True)
run_planner = st.sidebar.button("🚀 Run Path Planner", type="primary")

with st.expander("📋 Current Configuration"):
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.write("**Agent**")
        st.write(f"({agent_x:.1f}, {agent_y:.1f}), heading={agent_heading}°, speed={agent_speed:.1f}")
    with colB:
        st.write("**Intruder**")
        st.write(f"({intruder_x:.1f}, {intruder_y:.1f}), speed={intruder_speed:.1f}")
        st.write(f"move={move_intruder}, index_speed={index_speed:.2f}")
    with colC:
        st.write("**EZ**")
        st.write(f"μ={mu:.3f}, R={R_ez:.1f}, r={r:.2f}")
    with colD:
        st.write("**Goal / Env**")
        st.write(f"goal=({goal_x:.1f}, {goal_y:.1f})")
        st.write(f"{workspace_mode}, wind={use_wind}")

# ============================
# Run planner
# ============================
if run_planner:
    xi_values, ez_points, _ = rho_values(mu, R_ez, r)
    if xi_values is None or ez_points is None or len(xi_values) == 0 or len(ez_points) == 0:
        st.error("❌ Cannot compute EZ boundary. Invalid parameters.")
    else:
        start = [agent_x, agent_y]
        goal = [goal_x, goal_y]

        # Compute intruder heading toward agent start (main.py)
        heading_to_start_deg = 0.0
        if move_intruder:
            dx = start[0] - float(intruder_x)
            dy = start[1] - float(intruder_y)
            heading_to_start_deg = math.degrees(math.atan2(dy, dx))
            intruder_parameters["heading_deg"] = heading_to_start_deg
            intruder_parameters["index_speed"] = float(index_speed)

        # Compute EZ circle obstacle (main.py)
        ez_arr = np.asarray(ez_points, dtype=float)
        xi_arr = np.asarray(xi_values, dtype=float)
        bx = ez_arr * np.cos(xi_arr)
        by = ez_arr * np.sin(xi_arr)
        rel_cx, rel_cy, ez_radius = fit_circle(bx, by)

        ez_center = (float(intruder_x) + rel_cx, float(intruder_y) + rel_cy)
        obstacles = [(ez_center[0], ez_center[1], ez_radius)]

        # Workspace (fixed map_size like main.py OR auto-fit)
        map_size = [50, 50] if fixed_map else None
        shifted_start, shifted_goal, shifted_obstacles, map_size, shift_x, shift_y = compute_workspace(
            start, goal, obstacles, map_size=map_size
        )

        # Intruder shifted parameters (main.py style)
        intruder_shifted = {
            "x": float(intruder_x) + shift_x,
            "y": float(intruder_y) + shift_y,
            "speed": float(intruder_speed),
            "heading_deg": float(intruder_parameters.get("heading_deg", 0.0)),
            "index_speed": float(intruder_parameters.get("index_speed", 0.0)),
        }

        wind_path_arg = wind_data if (use_wind and wind_data.exists()) else None

        with st.spinner("🔄 Planning optimal path..."):
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
                max_iter=int(max_iter),
                edge_sample_points=int(edge_sample_points),
                wind_csv_path=wind_path_arg,   # <-- main.py feature
            )

            path = rrt.plan()
            elapsed = time.perf_counter() - start_time

        # Save in session state
        st.session_state.rrt = rrt
        st.session_state.path = path
        st.session_state.elapsed = elapsed
        st.session_state.shift_x = shift_x
        st.session_state.shift_y = shift_y
        st.session_state.intruder_shifted = intruder_shifted
        st.session_state.xi_values = xi_values
        st.session_state.ez_points = ez_points
        st.session_state.intruder_world = (float(intruder_x), float(intruder_y))
        st.session_state.R_ez = float(R_ez)
        st.session_state.r = float(r)

        if path:
            st.success(f"✅ Path found in {elapsed:.2f}s | nodes={len(rrt.node_list):,} | waypoints={len(path)}")
        else:
            st.error(f"❌ No path found in {elapsed:.2f}s | nodes={len(rrt.node_list):,}")

        # Export to output/simulations/ (matches plot_results.py expectations)
        if export_sim and path:
            repo_root = APP_ROOT.parent  # Go up from /code to repo root
            out_dir = (repo_root / "output" / "simulations").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            rrt.export_simulation_csvs(
                path=path,
                output_dir=out_dir,
                dt=1.0,
                n_theta=6,
                run_id="streamlit_run",
            )
            st.session_state.exported_to = out_dir
            st.info(f"📁 Exported simulation CSVs to: {out_dir}")

# ============================
# Results
# ============================
if "rrt" in st.session_state:
    rrt = st.session_state.rrt
    path = st.session_state.path

    st.divider()
    st.subheader("📊 Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Planning Time", f"{st.session_state.elapsed:.2f}s")
    c2.metric("Tree Nodes", f"{len(rrt.node_list):,}")
    c3.metric("Path", "✅ Found" if path else "❌ Not found")

   # ============================
    # Simplified Visualization (3 Tabs)
    # ============================
    # Merged "Overview" & "Paper Figure" into one "Trajectory" tab to remove redundancy.
    # The high-quality "Paper Figure" is now the default view.
    tabs = st.tabs(["📍 Trajectory", "🎬 Animation", "📊 Diagnostics"])

    # ---------- 1. Trajectory (High Quality Static Map) ----------
    with tabs[0]:
        if path and ("exported_to" in st.session_state):
            try:
                # Disable frame saving for speed
                pr.SAVE_ANIMATION_FRAMES = False
                agent_data, intruder_data, ez_data, wind_data = pr.load_results()
                
                # Plot the high-res figure (Paper Figure)
                # This is better than the old "Overview" because it shows the wind heat map
                fig, _ = pr.plot_final_paper_figure(agent_data, intruder_data, ez_data, wind_data)
                
                st.pyplot(fig, width='stretch')
                plt.close(fig)
                
                st.caption("Visualization of the optimal path over the wind uncertainty map.")
            
            except Exception as e:
                st.error(f"Could not generate detailed map: {e}")
                # Fallback to simple plot if the complex one fails
                fig, _ = rrt.plot(
                    intruder=(st.session_state.intruder_shifted["x"], st.session_state.intruder_shifted["y"]),
                    R=st.session_state.R_ez,
                    r=st.session_state.r,
                    origin_shift=(st.session_state.shift_x, st.session_state.shift_y),
                )
                st.pyplot(fig, width='stretch')
                plt.close(fig)
        
        elif path:
             st.info("⚠️ Export simulation CSVs to view the detailed heat map.")
        else:
            st.warning("No path found.")

    # ---------- 2. Animation ----------
    with tabs[1]:
        if path and ("exported_to" in st.session_state):
            
            # # Compact Controls
            # c1, c2 = st.columns([1, 2])
            # fps = c1.slider("Speed (FPS)", 1, 20, 5)
            # embed_frames = c2.checkbox("Embed frames (for saving HTML)", value=False)

            with st.spinner("Generating animation..."):
                try:
                    pr.SAVE_ANIMATION_FRAMES = False
                    agent_data, intruder_data, ez_data, wind_data = pr.load_results()

                    # Use low DPI (80) so it fits nicely on web screens
                    try:
                        fig, ani = pr.plot_with_wind_animation(
                            agent_data, intruder_data, ez_data, wind_data,
                            figsize=(10, 8), dpi=80
                        )
                    except TypeError:
                        fig, ani = pr.plot_with_wind_animation(agent_data, intruder_data, ez_data, wind_data)

                    # html_str = ani.to_jshtml(fps=fps, embed_frames=embed_frames, default_mode="loop")
                    # plt.close(fig)
                    
                    html_str = ani.to_jshtml(fps=5, embed_frames=True, default_mode="loop")
                    
                    # Save animation as GIF using temporary file
                    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    writer = PillowWriter(fps=5)
                    ani.save(tmp_path, writer=writer)
                    
                    # Read the GIF file into BytesIO
                    with open(tmp_path, "rb") as f:
                        gif_buffer = BytesIO(f.read())
                    
                    # Clean up temporary file
                    Path(tmp_path).unlink()
                    plt.close(fig)

                    # Display animation
                    # CSS for automatic width adjustment
                    style_block = """
                    <style>
                        .anim-container { width: 100%; display: flex; justify-content: center; }
                        .anim-container video, .anim-container img { width: 100% !important; height: auto !important; }
                    </style>
                    """
                    components.html(f"{style_block}<div class='anim-container'>{html_str}</div>", height=850, scrolling=False)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Animation as GIF",
                        data=gif_buffer.getvalue(),
                        file_name="path_animation.gif",
                        mime="image/gif",
                    )

                except Exception as e:
                    st.error(f"Animation failed: {e}")
        else:
            st.info("You need to run the planner with 'Export simulation CSVs' enabled.")

    # ---------- 3. Diagnostics ----------
    with tabs[2]:
        if path:
            st.markdown("#### 📉 Safety Metrics")
            # Calculate detailed metrics
            unshifted = [[p[0] - st.session_state.shift_x, p[1] - st.session_state.shift_y] for p in path]
            diagnostics = compute_path_diagnostics(
                unshifted,
                st.session_state.intruder_world,
                st.session_state.xi_values,
                st.session_state.ez_points,
            )

            if diagnostics is not None and not diagnostics.empty:
                metrics = summarize_path_metrics(diagnostics)
                
                # Cleaner data visualization
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Length", f"{metrics['path_length']:.2f}")
                m2.metric("Mean Clearance", f"{metrics['mean_clearance']:.3f}")
                m3.metric("Min Clearance", f"{metrics['min_clearance']:.3f}", delta_color="normal")
                m4.metric("Safe Points", f"{metrics['percent_safe_points']:.1f}%")

                st.divider()
                st.caption("Point-by-point path details:")
                st.dataframe(diagnostics, width='stretch', height=250)

                st.download_button(
                    "📥 Download Diagnostics CSV",
                    data=diagnostics.to_csv(index=False),
                    file_name="path_diagnostics.csv",
                    mime="text/csv",
                )
        else:
            st.info("No path data available for analysis.")

# # ============================
# # Additional Visualization Options
# # ============================
# if ("rrt" in st.session_state) and st.session_state.path and ("exported_to" in st.session_state):
#     st.divider()
#     st.subheader("📈 Additional Visualizations")
    
#     with st.expander("🎨 View Alternative Plots"):
#         try:
#             # Disable frame saving to avoid spam
#             pr.SAVE_ANIMATION_FRAMES = False
            
#             # Load exported CSVs
#             agent_data, intruder_data, ez_data, wind_data = pr.load_results()
            
#             # Create tabs for different visualizations
#             tab1, tab2 = st.tabs(["📊 Static Plot", "📄 Paper Figure"])
            
#             with tab1:
#                 st.info("Quick static visualization of the path")
#                 fig, _ = pr.plot_path(agent_data, intruder_data, ez_data)
#                 st.pyplot(fig, width='stretch')
#                 plt.close(fig)
            
#             with tab2:
#                 st.info("Publication-ready figure with wind uncertainty background")
#                 fig, _ = pr.plot_final_paper_figure(agent_data, intruder_data, ez_data, wind_data)
#                 st.pyplot(fig, width='stretch')
#                 plt.close(fig)
        
#         except Exception as e:
#             st.error(f"❌ Could not generate additional plots: {e}")
#             import traceback
#             st.code(traceback.format_exc())