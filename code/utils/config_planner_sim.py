import numpy as np
import math
import pandas as pd
from pathlib import Path
import csv
import json
from datetime import datetime

from utils.rrt_path_planner import RRTStar
from utils.engament_zone_model import agent, intruder, rho_values, wrap_angle_deg
from utils.config_planner_sim import *

OUTPUT_DIR = Path('../wind-rrt-ez/output')
IMAGES_DIR = OUTPUT_DIR / 'images'
SIM_OUTPUT_DIR = OUTPUT_DIR / 'simulations'
MISC_OUTPUT_DIR = OUTPUT_DIR / 'misc'
RUN_OUTPUT_DIR = SIM_OUTPUT_DIR

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


def compute_workspace(start, goal, obstacles, padding=0.15, map_size=None):
    """
    If map_size is None:
        - auto-compute workspace bounds and shift everything so it fits in [0, map_span]
    If map_size is provided:
        - do not shift anything (assumes inputs already match the environment grid)
    """
    if map_size is not None:
        # No shifting when map_size is explicitly provided
        shift_x, shift_y = 0.0, 0.0
        return start, goal, obstacles, list(map_size), shift_x, shift_y

    # Auto workspace computation (only if map_size is None)
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
    xi_wrapped = ((angle_rad + np.pi) % (2.0 * np.pi)) - np.pi
    xi_arr = np.asarray(xi_values, dtype=float)
    rho_arr = np.asarray(ez_points, dtype=float)
    idx = int(np.argmin(np.abs(xi_arr - xi_wrapped)))
    return float(rho_arr[idx])


def compute_path_diagnostics(path_points, intruder_pos, xi_values, ez_points):
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


def save_world_path(path, shift_x, shift_y, intruder_world, xi_values, ez_points, metadata=None):
    if not path:
        return None, None

    SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unshifted = [[p[0] - shift_x, p[1] - shift_y] for p in path]

    diagnostics = compute_path_diagnostics(unshifted, intruder_world, xi_values, ez_points)
    if diagnostics is None:
        return None, None

    path_csv = SIM_OUTPUT_DIR / 'rrt_path_world_coords.csv'
    diagnostics.to_csv(path_csv, index=False)

    summary = summarize_path_metrics(diagnostics)
    run_id = datetime.utcnow().isoformat()
    summary['run_id'] = run_id
    if metadata:
        summary.update(metadata)

    MISC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = MISC_OUTPUT_DIR / 'rrt_results_summary.csv'
    fieldnames = list(summary.keys())
    write_header = not summary_path.exists()
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    summary_txt_path = MISC_OUTPUT_DIR / 'rrt_results_summary_readable.txt'
    with open(summary_txt_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variable', 'value'])
        for key, value in summary.items():
            writer.writerow([key, value])

    config_path = MISC_OUTPUT_DIR / 'run_config.json'
    config_payload = metadata.copy() if metadata else {}
    config_payload['run_id'] = run_id
    config_payload['generated_at_utc'] = run_id
    config_payload['path_csv'] = str(path_csv)
    with open(config_path, 'w') as f:
        json.dump(config_payload, f, indent=2)

    return path_csv, summary_path


def save_plot(fig):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGES_DIR / 'rrt_path.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    return out_path