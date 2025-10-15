import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Shared color palette for Engagement Zone visuals
EZ_COLORS = {
    "zone_fill": "#91c9f7",
    "zone_edge": "#1d4ed8",
    "intruder": "#111827",
    "agent": "#0f766e",
    "agent_heading": "#0f766e",
    "range_circle": "#1e293b",
    "capture_circle": "#f97316",
    "capture_alpha": 0.65,
    "safe_fill": "#bbf7d0",
    "unsafe_fill": "#fecdd3",
    "line_of_sight": "#6366f1",
    "boundary_marker": "#dc2626",
    "boundary_marker_style": "X",
    "boundary_marker_size": 120,
    "info_box_face": "#f8fafc",
    "aspect_arc": "#0ea5e9",
    "aspect_text": "#0f172a",
}

def wrap_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees to the [-180, 180] range, including -180 and 180 only once."""
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    # If input is exactly 180 or -180, preserve sign
    if np.isclose(np.abs(angle) % 360, 180.0):
        return np.sign(angle) * 180.0
    return wrapped


def _calc_centered_limits(
    center_x: float,
    center_y: float,
    x_values,
    y_values,
    *,
    default_radius: float,
    padding_ratio: float = 0.08,
):
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    rel_x = np.abs(x_arr - center_x)
    rel_y = np.abs(y_arr - center_y)
    max_rel_x = rel_x.max() if rel_x.size else 0.0
    max_rel_y = rel_y.max() if rel_y.size else 0.0
    max_radius = max(default_radius, max_rel_x, max_rel_y)
    radius = max_radius * (1.0 + padding_ratio)
    return (
        (center_x - radius, center_x + radius),
        (center_y - radius, center_y + radius),
    )

def plot_engagement_zone_boundaries(mu, R, r, intruder_parameters, xi_values, ez_points, save_path=None, figsize=(12, 10)):
    intruder_x = intruder_parameters.get('x')
    intruder_y = intruder_parameters.get('y')
    intruder_speed = intruder_parameters.get('speed')
    
    x = intruder_x + ez_points * np.cos(xi_values)
    y = intruder_y + ez_points * np.sin(xi_values)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_facecolor('#f3f4f6')
    ez_area = ax.fill(
        x,
        y,
        color=EZ_COLORS["zone_fill"],
        alpha=0.55,
        label="Engagement Zone",
    )
    ax.plot(x, y, color=EZ_COLORS["zone_edge"], linewidth=2.2)
    intruder_plot = ax.scatter(
        intruder_x,
        intruder_y,
        marker="^",
        s=150,
        color=EZ_COLORS["intruder"],
        label="Intruder",
        zorder=6,
    )
    ax.annotate(
        f"Intruder\n({intruder_x:.2f}, {intruder_y:.2f})",
        (intruder_x, intruder_y + 0.18),
        xytext=(0, 0),
        textcoords='offset points',
        fontsize=10,
        ha='center',
        va='center',
        bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.92, edgecolor='#d1d5db', boxstyle='round,pad=0.5'),
    )

    # Add circles for R and r around the intruder
    circle_R = plt.Circle(
        (intruder_x, intruder_y),
        R,
        color=EZ_COLORS["range_circle"],
        linestyle="--",
        linewidth=2,
        fill=False,
        alpha=0.65,
        label=f"Pursuer Range (R={R})",
    )
    circle_r = plt.Circle(
        (intruder_x, intruder_y),
        r,
        color=EZ_COLORS["capture_circle"],
        linestyle=":",
        linewidth=2,
        fill=False,
        alpha=EZ_COLORS["capture_alpha"],
        label=f"Neutralization Radius (r={r})",
    )
    ax.add_patch(circle_R)
    ax.add_patch(circle_r)

    default_radius = max(R + r, ez_points.max()) * 1.2
    x_limits, y_limits = _calc_centered_limits(
        intruder_x,
        intruder_y,
        np.concatenate([x, [intruder_x]]),
        np.concatenate([y, [intruder_y]]),
        default_radius=default_radius,
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_title("Engagement Zone Boundaries", fontsize=16, pad=18)
    ax.set_xlabel("X", fontsize=13)
    ax.set_ylabel("Y", fontsize=13)
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=14, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=9, length=3)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=EZ_COLORS['intruder'], markersize=9, label=f'Intruder (x={intruder_x}, y={intruder_y})'),
        Line2D([0], [0], color=EZ_COLORS['zone_edge'], lw=2.2, linestyle='-', label='EZ Boundary'),
        Patch(facecolor=EZ_COLORS['zone_fill'], edgecolor='none', alpha=0.55, label='Engagement Zone'),
        Patch(facecolor='none', edgecolor=EZ_COLORS['range_circle'], linestyle='--', linewidth=2, label=f'Pursuer Range (R={R})'),
        Patch(facecolor='none', edgecolor=EZ_COLORS['capture_circle'], linestyle=':', linewidth=2, label=f'Neutralization Radius (r={r})'),
    ]
    ax.legend(handles=legend_handles, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.25), 
              frameon=True, fancybox=True, 
              ncols=3, fontsize=10, 
              borderpad=1.5, 
              columnspacing=2.0, 
              handletextpad=1.5)

    ax.text(
        0.02,
        0.02,
        f"Parameters: μ = {mu}, R = {R}, r = {r}",
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        verticalalignment="bottom",
        bbox=dict(facecolor=EZ_COLORS["info_box_face"], alpha=0.92, edgecolor="#d1d5db", boxstyle="round,pad=0.4"),
    )

    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.10, right=0.98)
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig

def plot_engagement_zone_interaction(ez_parameters, rho_values, save_path=None, figsize=(12, 6)):
    
    # Use only agent_safety output variables for all geometric logic
    mu = ez_parameters['mu']
    R = ez_parameters['R']
    r = ez_parameters['r']
    intruder_x = ez_parameters['intruder_x']
    intruder_y = ez_parameters['intruder_y']
    agent_x = ez_parameters['agent_x']
    agent_y = ez_parameters['agent_y']
    heading_deg = ez_parameters['heading_deg']
    aspect_deg = ez_parameters['aspect_deg']
    distance = ez_parameters['los']
    ez_radius = ez_parameters['rho']
    theta_LOS = ez_parameters['los_deg']
    boundary_x = ez_parameters['boundary_x']
    boundary_y = ez_parameters['boundary_y']
    # For plotting boundary, use rho_values
    xi_values, ez_points, _ = rho_values

    def _arc_bounds(start_deg, delta_deg):
        wrapped = wrap_angle_deg(delta_deg)
        if wrapped >= 0:
            return start_deg, start_deg + wrapped
        else:
            return start_deg + wrapped, start_deg

    x_boundary = intruder_x + ez_points * np.cos(xi_values)
    y_boundary = intruder_y + ez_points * np.sin(xi_values)

    # Use pre-computed safety results from ez_parameters
    safe = ez_parameters['safe']
    message = ez_parameters['message']
    fill_color = EZ_COLORS["safe_fill"] if safe else EZ_COLORS["unsafe_fill"]
    agent_heading_rad = np.deg2rad(heading_deg)
    

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_facecolor('#f3f4f6')
    ax.fill(x_boundary, y_boundary, color=fill_color, alpha=0.55)
    ax.plot(x_boundary, y_boundary, color=EZ_COLORS['zone_edge'], linewidth=2.2)

    circle_R = plt.Circle((intruder_x, intruder_y), R, color=EZ_COLORS['range_circle'], linestyle='--', linewidth=2, fill=False, alpha=0.65, label=f'Pursuer Range (R={R})')
    circle_r = plt.Circle((intruder_x, intruder_y), r, color=EZ_COLORS['capture_circle'], linestyle=':', linewidth=2, fill=False, alpha=EZ_COLORS['capture_alpha'], label=f'Neutralization Radius (r={r})')
    ax.add_patch(circle_R)
    ax.add_patch(circle_r)

    intruder_plot = ax.scatter(intruder_x, intruder_y, marker='^', s=150, color=EZ_COLORS['intruder'], label='Intruder', zorder=6)
    ax.annotate(
        f"Intruder\n({intruder_x:.2f}, {intruder_y:.2f})",
        (intruder_x, intruder_y + 0.18),
        xytext=(0, 0),
        textcoords='offset points',
        fontsize=10,
        ha='center',
        va='center',
        bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.92, edgecolor='#d1d5db', boxstyle='round,pad=0.5'),
    )
    agent_plot = ax.scatter(agent_x, agent_y, marker='o', s=160, color=EZ_COLORS['agent'], label='Agent', zorder=7)
    ax.annotate(
        f"Agent\n({agent_x:.2f}, {agent_y:.2f})",
        (agent_x, agent_y - 0.18),
        xytext=(0, 0),
        textcoords='offset points',
        fontsize=10,
        ha='center',
        va='center',
        bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.92, edgecolor='#d1d5db', boxstyle='round,pad=0.5'),
    )

    fixed_arrow_length = max(R, 1.0) * 0.6
    heading_arrow = ax.annotate(
        '',
        xy=(agent_x + fixed_arrow_length * np.cos(agent_heading_rad), agent_y + fixed_arrow_length * np.sin(agent_heading_rad)),
        xytext=(agent_x, agent_y),
        arrowprops=dict(arrowstyle='->', color=EZ_COLORS['agent_heading'], lw=2.5),
        annotation_clip=False,
        label='Agent Heading'
    )
    # Remove heading angle annotation from the plot; keep only in legend

    los_line = ax.annotate(
        '',
        xy=(agent_x, agent_y),
        xytext=(intruder_x, intruder_y),
        arrowprops=dict(arrowstyle='->', color=EZ_COLORS['line_of_sight'], lw=2.5, linestyle='--'),
        annotation_clip=False,
        label='Line of Sight'
    )
    
    # ax.text(
    #     intruder_x + 0.35 * distance * np.cos(los_intruder),
    #     intruder_y + 0.35 * distance * np.sin(los_intruder),
    #     f"θ_LOS = {los_intruder_deg:.1f}°",
    #     fontsize=9,
    #     ha='center',
    #     va='center',
    #     color=EZ_COLORS['aspect_text'],
    #     bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.7, edgecolor='none'),
    # )

    # EZ boundary marker at correct border using aspect_deg
    # aspect_rad = np.deg2rad(aspect_deg)
    # ez_boundary_x = intruder_x + ez_radius * np.cos(aspect_rad)
    # ez_boundary_y = intruder_y + ez_radius * np.sin(aspect_rad)
    # boundary_point = ax.scatter(
    #     ez_boundary_x,
    #     ez_boundary_y,
    #     color=EZ_COLORS['boundary_marker'],
    #     marker=EZ_COLORS['boundary_marker_style'],
    #     s=EZ_COLORS['boundary_marker_size'],
    #     label='EZ Boundary',
    #     zorder=7,
    # )

    # Draw LOS and heading reference lines with consistent styling
    ax.plot(
        [intruder_x, agent_x],
        [intruder_y, agent_y],
        color=EZ_COLORS['line_of_sight'],
        lw=2.3,
        linestyle=':',
        alpha=0.9,  # Shorter dashes for LOS
        zorder=3,
    )
    ax.plot(
        [agent_x, agent_x + fixed_arrow_length * np.cos(agent_heading_rad)],
        [agent_y, agent_y + fixed_arrow_length * np.sin(agent_heading_rad)],
        color=EZ_COLORS['agent_heading'],
        lw=2.3,
        linestyle='-',
        alpha=0.95,
        zorder=5,
    )

    # --- θ_LOS angle arc (from intruder reference to LOS direction) ---
    # Make arc radius larger and respect angle signs properly
    aspect_radius = max(0.6 * R, min(distance * 0.4, R * 0.8))
    psi_intruder = 0.0  # Intruder reference direction is 0° (positive X-axis)
    
    # Draw arc from reference direction (0°) to actual LOS angle (with proper sign)
    # θ_LOS is already correctly calculated with sign in agent_safety function
    if theta_LOS >= 0:
        # Positive angle: arc goes counter-clockwise from 0° to +theta_LOS
        theta1_aspect, theta2_aspect = psi_intruder, theta_LOS
    else:
        # Negative angle: arc goes clockwise from theta_LOS to 0°
        theta1_aspect, theta2_aspect = theta_LOS, psi_intruder
    
    aspect_arc = Arc(
        (intruder_x, intruder_y),
        width=2 * aspect_radius,
        height=2 * aspect_radius,
        angle=0,
        theta1=theta1_aspect,
        theta2=theta2_aspect,
        color=EZ_COLORS['aspect_arc'],
        linewidth=2.8,
        zorder=6,
    )
    ax.add_patch(aspect_arc)
    
    # Position label at the arc midpoint, preserving the sign
    aspect_mid_angle = np.deg2rad((theta1_aspect + theta2_aspect) / 2.0)
    label_radius = aspect_radius * 1.2
    aspect_label = ax.text(
        intruder_x + label_radius * np.cos(aspect_mid_angle),
        intruder_y + label_radius * np.sin(aspect_mid_angle),
        f"θ = {theta_LOS:.1f}°",
        fontsize=10,
        ha='center',
        va='center',
        color=EZ_COLORS['aspect_arc'],
        bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.85, edgecolor='#d1d5db', boxstyle='round,pad=0.4'),
    )

    # --- Aspect angle arc (ξ) from LOS direction to agent heading (clockwise convention) ---
    psi_agent = heading_deg
    rel_arc_handle = None
    if abs(aspect_deg) > 1e-2 and distance > 0:
        # aspect_deg is the clockwise angle from LOS direction to agent heading
        # Positive aspect: clockwise rotation from LOS to heading
        # Negative aspect: counter-clockwise rotation from LOS to heading (heading "above" LOS)
        los_direction = theta_LOS  # LOS direction from intruder to agent
        
        if aspect_deg >= 0:
            # Positive aspect: clockwise from LOS to heading
            # Arc should go counter-clockwise from LOS to heading for matplotlib
            theta1_rel, theta2_rel = los_direction, psi_agent
        else:
            # Negative aspect: counter-clockwise from LOS to heading  
            # Arc should go counter-clockwise from heading to LOS for matplotlib
            theta1_rel, theta2_rel = psi_agent, los_direction
        
        rel_radius = min(distance * 0.5, fixed_arrow_length * 0.75)
        rel_arc = Arc(
            (agent_x, agent_y),
            width=2 * rel_radius,
            height=2 * rel_radius,
            angle=0,
            theta1=theta1_rel,
            theta2=theta2_rel,
            color='#22c55e',
            linewidth=2.8,
            zorder=6,
        )
        ax.add_patch(rel_arc)
        rel_mid_angle = np.deg2rad((theta1_rel + theta2_rel) / 2.0)
        rel_label_pos = (
            agent_x + 0.9 * rel_radius * np.cos(rel_mid_angle),
            agent_y + 0.9 * rel_radius * np.sin(rel_mid_angle),
        )
        rel_arc_handle = rel_arc
    else:
        # For very small aspect angles, position label near agent heading
        rel_label_pos = (
            agent_x + 0.35 * fixed_arrow_length * np.cos(agent_heading_rad + np.pi / 2),
            agent_y + 0.35 * fixed_arrow_length * np.sin(agent_heading_rad + np.pi / 2),
        )
        rel_arc_handle = Line2D([], [], color='#22c55e', linestyle='--', linewidth=2.0)

    rel_label = ax.text(
        rel_label_pos[0],
        rel_label_pos[1],
        f"ξ = {aspect_deg:.1f}°",
        fontsize=10,
        ha='center',
        va='center',
        color='#15803d',
        bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.85, edgecolor='#d1d5db', boxstyle='round,pad=0.4'),
    )

    relation = '≤' if distance <= ez_radius else '>'
    info_text = f"LOS = {distance:.2f} {relation} ρ = {ez_radius:.2f}"
    mid_x = (agent_x + intruder_x) / 2.0
    mid_y = (agent_y + intruder_y) / 2.0
    los_dx = agent_x - intruder_x
    los_dy = agent_y - intruder_y
    los_norm = np.hypot(los_dx, los_dy)
    los_angle_deg_display = np.degrees(np.arctan2(los_dy, los_dx)) if los_norm else 0.0
    offset_scale = 0.06 * max(R, r, distance, 1.0)
    offset_x = -los_dy / los_norm * offset_scale if los_norm else 0.0
    offset_y = los_dx / los_norm * offset_scale if los_norm else offset_scale
    ax.annotate(
        info_text,
        (mid_x + offset_x, mid_y + offset_y),
        ha='center',
        va='center',
        rotation=los_angle_deg_display,
        rotation_mode='anchor',
        fontsize=10,
        bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.85, edgecolor='#d1d5db', boxstyle='round,pad=0.4'),
    )

    default_radius = max(R + r, ez_points.max()) * 1.2
    arrow_end_x = agent_x + fixed_arrow_length * np.cos(agent_heading_rad)
    arrow_end_y = agent_y + fixed_arrow_length * np.sin(agent_heading_rad)
    bound_x_values = np.concatenate([
        x_boundary,
        [intruder_x, agent_x, arrow_end_x],
    ])
    bound_y_values = np.concatenate([
        y_boundary,
        [intruder_y, agent_y, arrow_end_y],
    ])
    x_limits, y_limits = _calc_centered_limits(
        intruder_x,
        intruder_y,
        bound_x_values,
        bound_y_values,
        default_radius=default_radius,
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_title(message, fontsize=16, pad=18)
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=14, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=9, length=3)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=EZ_COLORS['agent'], markersize=9, label=f'Agent (x={agent_x}, y={agent_y})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=EZ_COLORS['intruder'], markersize=9, label=f'Intruder (x={intruder_x}, y={intruder_y})'),
        Line2D([0], [0], color=EZ_COLORS['line_of_sight'], lw=2.3, linestyle=':', label=f'Line of Sight (LOS = {distance:.2f})'),
        Line2D([0], [0], color=EZ_COLORS['agent_heading'], lw=2.3, linestyle='-', label=f'Agent Heading (ψ = {heading_deg:.1f}°)'),
        Line2D([0], [0], color=EZ_COLORS['aspect_arc'], lw=2.3, linestyle='-', label=f'LOS Angle (θ = {theta_LOS:.1f}°)'),
        Line2D([0], [0], color='#22c55e', lw=2.3, linestyle='-', 
               label=f'Aspect Angle (ξ = {aspect_deg:.1f}°)'),
        # Line2D([0], [0], marker=EZ_COLORS['boundary_marker_style'], color=EZ_COLORS['boundary_marker'], linestyle='None', markersize=9, label=f'ρ (x,y) = ({ez_boundary_x:.2f}, {ez_boundary_y:.2f})'),
        Patch(facecolor='none', edgecolor=EZ_COLORS['range_circle'], linestyle='--', linewidth=2, label=f'Pursuer Range (R={R})'),
        Patch(facecolor='none', edgecolor=EZ_COLORS['capture_circle'], linestyle=':', linewidth=2, label=f'Neutralization Radius (r={r})'),
    ]
    ax.legend(handles=legend_handles, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.25), 
              frameon=True, fancybox=True, 
              ncols=3, fontsize=10, 
              borderpad=1.5, 
              columnspacing=2.0, 
              handletextpad=1.5)

    ax.text(
        0.02,
        0.02,
        f'Parameters: μ = {mu}, R = {R}, r = {r}',
        transform=ax.transAxes,
        fontsize=10,
        color='black',
        verticalalignment='bottom',
        bbox=dict(facecolor=EZ_COLORS['info_box_face'], alpha=0.92, edgecolor='#d1d5db', boxstyle='round,pad=0.4'),
    )

    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.10, right=0.98)
    if save_path:
        fig.savefig(save_path, dpi=300)
        summary_items = [
            ('Agent_x', agent_x),
            ('Agent_y', agent_y),
            ('Agent_heading_deg', heading_deg),
            ('Aspect_angle_deg', aspect_deg),
            ('LOS_angle_deg', theta_LOS),
            ('Agent_distance', distance),
            ('EZ_boundary_distance', ez_radius),
            ('Safety_status', 'outside EZ' if safe else 'inside EZ'),
            ('mu', mu),
            ('R', R),
            ('r', r),
        ]
        summary_df = pd.DataFrame(summary_items, columns=['variable', 'value'])
        csv_path = '../data/ez_interaction_summary.csv'
        txt_path = '../data/ez_interaction_summary.txt'
        summary_df.to_csv(csv_path, index=False)
        with open(txt_path, 'w') as handle:
            handle.write(summary_df.to_string(index=False))
    return fig