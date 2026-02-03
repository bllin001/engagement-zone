"""
Plot RRT* path planning results from CSV output files with wind uncertainty visualization and animation.
Publication-quality visualization with telemetry panel and professional styling.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import math

# --- Publication & Presentation Styling (light theme) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "grid.color": "#cccccc",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "axes.edgecolor": "#444444",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

# OUTPUT_DIR = Path('output/ez-rrt-path')
OUTPUT_DIR = Path('output')
SIM_OUTPUT_DIR = OUTPUT_DIR / 'simulations'
IMAGES_OUTPUT_DIR = OUTPUT_DIR / 'images'
FRAMES_OUTPUT_DIR = IMAGES_OUTPUT_DIR / 'frames'
WIND_DATA_PATH = Path('env/output/harvey_wind_probabilities_for_EZ.csv')
SHOW_WIND_VECTORS = False #True/False
WIND_VECTOR_STRIDE = 3
SAVE_ANIMATION_FRAMES = False  # Set to True to save individual frames (disabled by default for Streamlit)

def fit_circle(x_coords, y_coords):
    """
    Compute the center (a, b) and radius of a circle passing through the given points.
    Uses a simple least-squares fit which produces an exact result for ideal EZ data.
    """
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

def load_results():
    """Load agent, intruder, EZ boundary, and wind data with statistical calculations."""
    agent_csv = SIM_OUTPUT_DIR / 'agent_state.csv'
    intruder_csv = SIM_OUTPUT_DIR / 'intruder_state.csv'
    ez_csv = SIM_OUTPUT_DIR / 'ez_boundary.csv'
    
    agent_data = pd.read_csv(agent_csv)
    intruder_data = pd.read_csv(intruder_csv)
    ez_data = pd.read_csv(ez_csv)
    
    # Calculate expected speed in agent data for telemetry
    if 'wind_speed_1' in agent_data.columns and 'probability_1' in agent_data.columns:
        agent_data['wind_speed_2'] = agent_data['wind_speed_2'].fillna(0)
        agent_data['probability_2'] = agent_data['probability_2'].fillna(0)
        agent_data['expected_speed'] = (agent_data['wind_speed_1'] * agent_data['probability_1']) + \
                                       (agent_data['wind_speed_2'] * agent_data['probability_2'])
    
    # Build adaptive uncertainty field using Harvey datasets (summary + GMM + raw points)
    wind_data = None
    summary_path = Path('env/data/harvey_cell_summary.csv')
    dist_path = Path('env/data/harvey_distribution_types_gmm.csv')
    points_path = Path('env/data/harvey_all_points_with_cells.csv')

    try:
        if summary_path.exists() and dist_path.exists() and points_path.exists():
            df_summary = pd.read_csv(summary_path)
            df_dist = pd.read_csv(dist_path)
            df_summary = df_summary.merge(df_dist[['cell_id', 'distribution']], on='cell_id', how='left')

            df_points = pd.read_csv(points_path)
            from collections import defaultdict
            cell_speeds = defaultdict(list)
            for _, row in df_points.iterrows():
                cell_speeds[row['cell_id']].append(row['WIND_SPEED'])

            rows = []
            for _, row in df_summary.iterrows():
                cell_id = int(row['cell_id'])
                lon_idx = int(row['lon_idx'])
                lat_idx = int(row['lat_idx'])
                mean_speed = row['mean_speed']
                mean_u = row['mean_u']
                mean_v = row['mean_v']
                dist_type = row.get('distribution', 'unimodal')

                speeds = np.array(cell_speeds[cell_id], dtype=float)
                n = len(speeds)
                uncertainty = row['std_speed']

                if dist_type == 'bimodal' and n >= 2:
                    mean_val = np.mean(speeds)
                    high_mask = speeds > mean_val
                    low_mask = speeds <= mean_val
                    n_high = np.sum(high_mask)
                    n_low = np.sum(low_mask)

                    if n_high > 0 and n_low > 0:
                        p_high = n_high / n
                        p_low = n_low / n
                        mu_high = np.mean(speeds[high_mask])
                        mu_low = np.mean(speeds[low_mask])
                        sigma_high_sq = np.var(speeds[high_mask], ddof=0) if n_high > 1 else 0.0
                        sigma_low_sq = np.var(speeds[low_mask], ddof=0) if n_low > 1 else 0.0

                        mean_mixture = p_high * mu_high + p_low * mu_low
                        variance_mixture = (
                            p_high * (sigma_high_sq + mu_high ** 2)
                            + p_low * (sigma_low_sq + mu_low ** 2)
                            - mean_mixture ** 2
                        )
                        uncertainty = math.sqrt(max(0.0, variance_mixture))

                rows.append(
                    {
                        'lon_idx': lon_idx,
                        'lat_idx': lat_idx,
                        'std_speed': uncertainty,
                        'mean_speed': mean_speed,
                        'mean_u': mean_u,
                        'mean_v': mean_v,
                    }
                )

            wind_data = pd.DataFrame(rows)
        elif WIND_DATA_PATH.exists():
            # Fallback to provided probability file if adaptive sources are missing
            wind_data = pd.read_csv(WIND_DATA_PATH)
            wind_data['wind_speed_2'] = wind_data['wind_speed_2'].fillna(0)
            wind_data['probability_2'] = wind_data['probability_2'].fillna(0)
            wind_data['expected_speed'] = (wind_data['wind_speed_1'] * wind_data['probability_1']) + (
                wind_data['wind_speed_2'] * wind_data['probability_2']
            )
            wind_data['variance_speed'] = (
                wind_data['probability_1'] * (wind_data['wind_speed_1'] - wind_data['expected_speed']) ** 2
            ) + (
                wind_data['probability_2'] * (wind_data['wind_speed_2'] - wind_data['expected_speed']) ** 2
            )
            wind_data['std_speed'] = np.sqrt(wind_data['variance_speed'])
            wind_data['uncertainty'] = wind_data['probability_1'] * wind_data['probability_2']
    except Exception as e:
        print(f"Warning: could not compute adaptive uncertainty field: {e}")
    
    return agent_data, intruder_data, ez_data, wind_data

def plot_path(agent_data, intruder_data, ez_data):
    """Plot the RRT* path with EZ boundary and intruder."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract agent path
    agent_x = agent_data['lon_idx'].values
    agent_y = agent_data['lat_idx'].values
    
    # Plot agent path
    ax.plot(agent_x, agent_y, color='#0b5573', linewidth=3.2, label='Final Path', zorder=5)
    ax.scatter(agent_x[0], agent_y[0], color='#1b7ccc', s=150, marker='*', 
               label=f'Start ({agent_x[0]:.1f}, {agent_y[0]:.1f})', zorder=6, edgecolors='white', linewidth=1.1)
    ax.scatter(agent_x[-1], agent_y[-1], color='#ffc857', s=160, marker='o',
               label=f'Goal ({agent_x[-1]:.1f}, {agent_y[-1]:.1f})', zorder=6, edgecolors='white', linewidth=1.1)
    
    # Fit circle to EZ boundary points
    ez_boundary_x = ez_data['boundary_lon_idx'].values
    ez_boundary_y = ez_data['boundary_lat_idx'].values
    center_x, center_y, radius = fit_circle(ez_boundary_x, ez_boundary_y)
    
    # Plot EZ boundary as a proper circle
    circle = plt.Circle((center_x, center_y), radius, 
                        fill=True, alpha=0.16, color='#c23b2a', label='EZ (Engagement Zone)', zorder=2)
    ax.add_patch(circle)
    circle_outline = plt.Circle((center_x, center_y), radius, 
                               fill=False, color='#8b1e1e', linewidth=2.2, alpha=0.85, zorder=3)
    ax.add_patch(circle_outline)
    
    # Plot intruder
    intruder_x = intruder_data['lon_idx'].iloc[0]
    intruder_y = intruder_data['lat_idx'].iloc[0]
    ax.scatter(intruder_x, intruder_y, color='black', s=250, marker='*',
               label=f'Intruder ({intruder_x:.1f}, {intruder_y:.1f})', zorder=5, edgecolors='gray', linewidth=1)
    
    ax.set_xlabel('Longitude Idx', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude Idx', fontsize=12, fontweight='bold')
    ax.set_title('RRT* Path Planning', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(-0.5, 49.5)
    leg = ax.legend(loc='upper left', fontsize=10, framealpha=0.85)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#dddddd')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    return fig, ax


def plot_with_wind_animation(agent_data, intruder_data, ez_data, wind_data):
    """Create professional animation with wind heatmap, uncertainty contours, and telemetry panel."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1])
    ax = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis('off')

    # Prepare wind data grids if available
    if wind_data is not None and not wind_data.empty:
        try:
            # Ensure std_speed exists (fallback to mixture variance if missing)
            if 'std_speed' not in wind_data.columns:
                if {'wind_speed_1', 'wind_speed_2', 'probability_1', 'probability_2'}.issubset(wind_data.columns):
                    wind_data = wind_data.copy()
                    wind_data['wind_speed_2'] = wind_data['wind_speed_2'].fillna(0)
                    wind_data['probability_2'] = wind_data['probability_2'].fillna(0)
                    wind_data['expected_speed'] = (wind_data['wind_speed_1'] * wind_data['probability_1']) + (
                        wind_data['wind_speed_2'] * wind_data['probability_2']
                    )
                    wind_data['variance_speed'] = (
                        wind_data['probability_1'] * (wind_data['wind_speed_1'] - wind_data['expected_speed']) ** 2
                    ) + (
                        wind_data['probability_2'] * (wind_data['wind_speed_2'] - wind_data['expected_speed']) ** 2
                    )
                    wind_data['std_speed'] = np.sqrt(wind_data['variance_speed'])

            # Create custom hurricane colormap
            colors = ['#00FFFF', '#00FF00', '#7FFF00', '#FFFF00', '#FFA500', '#FF4500', '#FF0000', '#8B0000', '#4B0082']
            cmap_hurricane = LinearSegmentedColormap.from_list('hurricane', colors, N=256)

            # Wind uncertainty background (Std[W])
            std_pivot = (
                wind_data
                .pivot_table(index='lat_idx', columns='lon_idx', values='std_speed')
                .sort_index(ascending=False)  # flip latitude to match 5-Uncertainty heatmap
                .values
            )

            # Set color scale based on data
            vmin_value = 0
            vmax_value = np.nanpercentile(std_pivot, 98)

            im = ax.imshow(
                std_pivot,
                origin='lower',
                cmap=cmap_hurricane,
                alpha=0.9,
                extent=[-0.5, 49.5, -0.5, 49.5],
                vmin=vmin_value,
                vmax=vmax_value
            )

            if SHOW_WIND_VECTORS and {'mean_u', 'mean_v', 'mean_speed'}.issubset(wind_data.columns):
                u_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_u')
                    .sort_index(ascending=False)
                    .values
                )
                v_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_v')
                    .sort_index(ascending=False)
                    .values
                )
                speed_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_speed')
                    .sort_index(ascending=False)
                    .values
                )
                arrow_scale = max(1.0, np.nanmax(speed_pivot) * 8)

                lon_grid = np.linspace(-0.5, 49.5, std_pivot.shape[1])
                lat_grid = np.linspace(-0.5, 49.5, std_pivot.shape[0])
                X, Y = np.meshgrid(lon_grid, lat_grid)
                ax.quiver(
                    X[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    Y[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    u_pivot[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    v_pivot[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    color='black', alpha=0.7, linewidth=0.6,
                    scale=arrow_scale, width=0.002, headwidth=3, headlength=3,
                )

            if SHOW_WIND_VECTORS and {'mean_u', 'mean_v', 'mean_speed'}.issubset(wind_data.columns):
                u_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_u')
                    .sort_index(ascending=False)
                    .values
                )
                v_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_v')
                    .sort_index(ascending=False)
                    .values
                )
                speed_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_speed')
                    .sort_index(ascending=False)
                    .values
                )
                arrow_scale = max(1.0, np.nanmax(speed_pivot) * 8)

                lon_grid = np.linspace(-0.5, 49.5, std_pivot.shape[1])
                lat_grid = np.linspace(-0.5, 49.5, std_pivot.shape[0])
                X, Y = np.meshgrid(lon_grid, lat_grid)
                ax.quiver(
                    X[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    Y[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    u_pivot[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    v_pivot[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    color='black', alpha=0.7, linewidth=0.6,
                    scale=arrow_scale, width=0.002, headwidth=3, headlength=3,
                )

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='#111111')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#111111')
            cbar.set_label('Wind Uncertainty σ [m/s]', rotation=270, labelpad=25, fontsize=14, fontweight='bold', color='#111111')
        except Exception as e:
            print(f"Warning: Could not create wind grid: {e}")
    
    # RRT* Path line
    path_line, = ax.plot([], [], color='#0b5573', linewidth=3.2, label='Optimal RRT* Path', zorder=10)
    
    # Agent plane marker with rotation capability
    agent_plane, = ax.plot([], [], marker=(3, 0, 0), markersize=15, color='white', 
                          markeredgecolor='#0b5573', markeredgewidth=2, label='Agent', zorder=11)
    
    # EZ boundary circles (static and outline)
    ez_patch = Circle((0, 0), 1, color='#c23b2a', alpha=0.16, label='Engagement Zone', zorder=5)
    ax.add_patch(ez_patch)
    ez_edge = Circle((0, 0), 1, color='#8b1e1e', fill=False, linewidth=2.2, linestyle='--', zorder=6)
    ax.add_patch(ez_edge)
    
    # Intruder marker (will move if intruder_data varies over t)
    intr_marker = ax.scatter([], [], color='#c23b2a', marker='X', s=150,
                             label='Target Intruder', edgecolors='#8b1e1e', linewidth=1.5, zorder=12)
    
    # Start and goal markers
    start_x = agent_data['lon_idx'].iloc[0]
    start_y = agent_data['lat_idx'].iloc[0]
    goal_x = agent_data['lon_idx'].iloc[-1]
    goal_y = agent_data['lat_idx'].iloc[-1]
    ax.scatter([start_x], [start_y], color='#1b7ccc', s=150, marker='*', 
              label=f'Start', zorder=5, edgecolors='white', linewidth=1.1)
    ax.scatter([goal_x], [goal_y], color='#ffc857', s=180, marker='D',
              label=f'Goal', zorder=5, edgecolors='white', linewidth=1.1)
    
    # Telemetry panel
    telemetry = ax_info.text(0.05, 0.9, '', transform=ax_info.transAxes, 
                            family='monospace', fontsize=10, verticalalignment='top',
                            color='#111111',
                            bbox=dict(boxstyle='round', facecolor='#f2f2f2', edgecolor='#999999', alpha=0.9))
    
    ax.set_xlabel('Longitude Idx', fontsize=12, fontweight='bold', color='#111111')
    ax.set_ylabel('Latitude Idx', fontsize=12, fontweight='bold', color='#111111')
    ax.set_title('Adaptive RRT* Navigation: Wind Uncertainty & Dynamic EZ', pad=20, fontsize=14, color='#111111')
    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(-0.5, 49.5)
    leg = ax.legend(loc='upper left', fontsize=10, framealpha=0.85)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#dddddd')
    
    ax.grid(True, alpha=0.4, linestyle='--', color='#cccccc')
    ax.set_aspect('equal')
    
    # Create frames directory if saving individual frames
    if SAVE_ANIMATION_FRAMES:
        FRAMES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Animation update function
    def update(frame):
        t_unique = sorted(agent_data['t'].unique())
        t = t_unique[frame]
        
        # Update agent path and position
        curr_agent = agent_data[agent_data['t'] <= t]
        if not curr_agent.empty:
            row = curr_agent.iloc[-1]
            path_line.set_data(curr_agent['lon_idx'].values, curr_agent['lat_idx'].values)
            agent_plane.set_data([row['lon_idx']], [row['lat_idx']])
            # Rotate plane marker based on heading (adjust offset if needed)
            agent_plane.set_marker((3, 0, -row['heading_deg'] + 90))
        
            # Update EZ boundary
            curr_ez = ez_data[ez_data['t'] == t]
            if not curr_ez.empty:
                ez_boundary_x = curr_ez['boundary_lon_idx'].values
                ez_boundary_y = curr_ez['boundary_lat_idx'].values
                try:
                    center_x, center_y, radius = fit_circle(ez_boundary_x, ez_boundary_y)
                    ez_patch.center = ez_edge.center = (center_x, center_y)
                    ez_patch.set_radius(radius)
                    ez_edge.set_radius(radius)
                    
                    # Safety metric: Distance to EZ boundary
                    dist_to_ez = math.sqrt((row['lon_idx'] - center_x)**2 + (row['lat_idx'] - center_y)**2) - radius
                except Exception:
                    dist_to_ez = 0

            # Update intruder position if available for this time
            curr_intr = intruder_data[intruder_data['t'] == t]
            if not curr_intr.empty:
                ix = curr_intr['lon_idx'].values[0]
                iy = curr_intr['lat_idx'].values[0]
                intr_marker.set_offsets([[ix, iy]])
            
            # Determine wind regime
            regime = "BIMODAL" if row['probability_2'] > 0.01 else "UNIMODAL"
            
            # Build telemetry text
            info_text = (
                f"--- MISSION STATUS ---\n\n"
                f"TIME:          {t:>6.1f}s\n"
                f"POSITION:      ({row['lon_idx']:>6.1f}, {row['lat_idx']:>6.1f})\n"
                f"HEADING:       {row['heading_deg']:>6.1f}°\n"
                f"AIRSPEED:      {row['agent_speed']:>6.1f} m/s\n\n"
                f"--- ENV ANALYSIS ---\n"
                f"REGIME:        {regime:>10s}\n"
                f"p(Scenario 1): {row['probability_1']:>8.2f}\n"
                f"p(Scenario 2): {row['probability_2']:>8.2f}\n"
                f"E[Wind]:       {row['expected_speed']:>6.2f} m/s\n\n"
                f"--- SAFETY METRIC ---\n"
                f"CLEARANCE:     {max(0, dist_to_ez):>6.2f} units\n"
                f"ZONE STATUS:   {'⚠ WARNING' if dist_to_ez < 5 else '✓ SAFE':>7s}"
            )
            telemetry.set_text(info_text)
        
        # Save individual frame if enabled
        if SAVE_ANIMATION_FRAMES:
            frame_path = FRAMES_OUTPUT_DIR / f"frame_{frame:03d}_t{t:.1f}s.png"
            fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        return path_line, agent_plane, ez_patch, ez_edge, intr_marker, telemetry
    
    # Create animation (interval in milliseconds)
    ani = animation.FuncAnimation(fig, update, frames=len(agent_data['t'].unique()), 
                                interval=500, blit=True, repeat=True)
    
    return fig, ani

def plot_final_paper_figure(agent_data, intruder_data, ez_data, wind_data):
    """Create publication figure with wind uncertainty background (no telemetry panel)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Prepare wind data background if available
    if wind_data is not None and not wind_data.empty:
        try:
            # Create custom hurricane colormap
            colors = ['#00FFFF', '#00FF00', '#7FFF00', '#FFFF00', '#FFA500', '#FF4500', '#FF0000', '#8B0000', '#4B0082']
            cmap_hurricane = LinearSegmentedColormap.from_list('hurricane', colors, N=256)
            
            std_pivot = (
                wind_data
                .pivot_table(index='lat_idx', columns='lon_idx', values='std_speed')
                .sort_index(ascending=False)  # flip latitude to match 5-Uncertainty heatmap
                .values
            )
            
            vmin_value = 0
            vmax_value = np.nanpercentile(std_pivot, 98)
            
            im = ax.imshow(
                std_pivot,
                origin='lower',
                cmap=cmap_hurricane,
                alpha=0.9,
                extent=[-0.5, 49.5, -0.5, 49.5],
                vmin=vmin_value,
                vmax=vmax_value
            )
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='#111111')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#111111')
            cbar.set_label('Wind Adaptive Uncertainty σ [m/s]', rotation=270, labelpad=25, fontsize=14, fontweight='bold', color='#111111')
            
            # Add wind vectors overlay
            if SHOW_WIND_VECTORS and {'mean_u', 'mean_v', 'mean_speed'}.issubset(wind_data.columns):
                u_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_u')
                    .sort_index(ascending=False)
                    .values
                )
                v_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_v')
                    .sort_index(ascending=False)
                    .values
                )
                speed_pivot = (
                    wind_data
                    .pivot_table(index='lat_idx', columns='lon_idx', values='mean_speed')
                    .sort_index(ascending=False)
                    .values
                )
                arrow_scale = max(1.0, np.nanmax(speed_pivot) * 8)

                lon_grid = np.linspace(-0.5, 49.5, std_pivot.shape[1])
                lat_grid = np.linspace(-0.5, 49.5, std_pivot.shape[0])
                X, Y = np.meshgrid(lon_grid, lat_grid)
                ax.quiver(
                    X[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    Y[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    u_pivot[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    v_pivot[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE],
                    color='black', alpha=0.7, linewidth=0.6,
                    scale=arrow_scale, width=0.002, headwidth=3, headlength=3,
                    zorder=8
                )
        except Exception as e:
            print(f"Warning: Could not create wind background: {e}")
    
    # Plot full RRT* path
    agent_x = agent_data['lon_idx'].values
    agent_y = agent_data['lat_idx'].values
    ax.plot(agent_x, agent_y, color='#0b5573', linewidth=3.2, label='Optimal RRT* Path', zorder=10)
    
    # Agent at final position
    ax.scatter([agent_x[-1]], [agent_y[-1]], marker=(3, 0, -agent_data['heading_deg'].iloc[-1] + 90), 
               s=200, color='white', edgecolors='#0b5573', linewidth=2, label='Agent', zorder=11)
    
    # EZ boundary at final timestep
    final_t = agent_data['t'].max()
    final_ez = ez_data[ez_data['t'] == final_t]
    if not final_ez.empty:
        ez_x = final_ez['boundary_lon_idx'].values
        ez_y = final_ez['boundary_lat_idx'].values
        center_x, center_y, radius = fit_circle(ez_x, ez_y)
        ez_patch = Circle((center_x, center_y), radius, color='#c23b2a', alpha=0.16, label='Engagement Zone', zorder=5)
        ax.add_patch(ez_patch)
        ez_edge = Circle((center_x, center_y), radius, color='#8b1e1e', fill=False, linewidth=2.2, linestyle='--', zorder=6)
        ax.add_patch(ez_edge)
    
    # Intruder
    intruder_x = intruder_data['lon_idx'].iloc[0]
    intruder_y = intruder_data['lat_idx'].iloc[0]
    ax.scatter(intruder_x, intruder_y, color='#c23b2a', marker='X', s=150, 
              label='Target Intruder', edgecolors='#8b1e1e', linewidth=1.5, zorder=12)
    
    # Start and goal
    start_x = agent_data['lon_idx'].iloc[0]
    start_y = agent_data['lat_idx'].iloc[0]
    goal_x = agent_data['lon_idx'].iloc[-1]
    goal_y = agent_data['lat_idx'].iloc[-1]
    ax.scatter([start_x], [start_y], color='#1b7ccc', s=150, marker='*', 
              label='Start', zorder=5, edgecolors='white', linewidth=1.1)
    ax.scatter([goal_x], [goal_y], color='#ffc857', s=180, marker='D',
              label='Goal', zorder=5, edgecolors='white', linewidth=1.1)
    
    ax.set_xlabel('Longitude Idx', fontsize=12, fontweight='bold', color='#111111')
    ax.set_ylabel('Latitude Idx', fontsize=12, fontweight='bold', color='#111111')
    ax.set_title('Adaptive RRT* Navigation: Wind Uncertainty & Dynamic EZ', pad=20, fontsize=14, color='#111111')
    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(-0.5, 49.5)
    leg = ax.legend(loc='upper left', fontsize=10, framealpha=0.85)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#dddddd')
    ax.grid(True, alpha=0.4, linestyle='--', color='#cccccc')
    ax.set_aspect('equal')
    
    return fig, ax

def calculate_path_distance(agent_data):
    """Calculate total Euclidean distance of the path."""
    x = agent_data['lon_idx'].values
    y = agent_data['lat_idx'].values
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(distances)

def main():
    # Check if output files exist
    if not OUTPUT_DIR.exists():
        print(f"Output directory not found: {OUTPUT_DIR}")
        return
    
    try:
        agent_data, intruder_data, ez_data, wind_data = load_results()
        
        # Calculate and display path distance
        path_distance = calculate_path_distance(agent_data)
        print(f"\n{'='*70}")
        print(f"OPTIMAL PATH DISTANCE: {path_distance:.2f} units")
        print(f"{'='*70}\n")
        
        # Create static plot
        print("Generating static plot...")
        fig1, ax = plot_path(agent_data, intruder_data, ez_data)
        IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = IMAGES_OUTPUT_DIR / 'rrt_path_plot.png'
        fig1.tight_layout()
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved static plot to {output_path}")
        
        # Create professional animated plot with wind data and telemetry
        print("Generating professional animated plot with telemetry...")
        fig2, ani = plot_with_wind_animation(agent_data, intruder_data, ez_data, wind_data)
        
        # Save animation as GIF
        try:
            gif_path = IMAGES_OUTPUT_DIR / 'rrt_path_animation.gif'
            ani.save(gif_path, writer='pillow', fps=5)
            print(f"Saved animation as GIF to {gif_path}")
            if SAVE_ANIMATION_FRAMES:
                num_frames = len(agent_data['t'].unique())
                print(f"Saved {num_frames} individual frames to {FRAMES_OUTPUT_DIR}/")
        except Exception as e:
            print(f"Could not save animation: {e}")
        
        # Save final paper figure (plot only, no telemetry)
        print("Generating final paper figure...")
        fig3, ax3 = plot_final_paper_figure(agent_data, intruder_data, ez_data, wind_data)
        paper_path = IMAGES_OUTPUT_DIR / 'final_rrt_path_paper.png'
        fig3.tight_layout()
        fig3.savefig(paper_path, dpi=300, bbox_inches='tight')
        print(f"Saved final paper figure to {paper_path}")
        
        plt.show()
    except Exception as e:
        print(f"Error loading or plotting results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
