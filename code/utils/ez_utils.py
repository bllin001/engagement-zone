import numpy as np
from typing import Iterable, List, Tuple, Optional


def build_obstacles_from_ez(
    ez_points,
    xi_values,
    intruder_x: float,
    intruder_y: float,
    obstacle_radius: Optional[float] = None,
    subsample: Optional[int] = None,
    mode: str = "per_point",
    hull_subsample: int = 1,
) -> List[Tuple[float, float, float]]:
    """
    Convert engagement-zone data to planner obstacles: list of (x,y,radius).

    - ez_points: either 1D radial distances (matching xi_values) OR
                 iterable of (x,y) pairs OR pandas DataFrame with 'x'/'y'.
    - xi_values: angles in radians (used only if ez_points are radial distances).
    - intruder_x/intruder_y: reference origin for radial data.
    - obstacle_radius: radius to assign to each obstacle. If None, caller should pass R.
    - subsample: if set, keep only every `subsample`-th point for 'per_point' mode.
    - mode: 'per_point' | 'single_circle' | 'hull_vertices'
    """
    points = []

    # Case A: radial distances + angles
    try:
        ez_arr = np.asarray(ez_points, dtype=float)
        xi_arr = np.asarray(xi_values, dtype=float)
        if ez_arr.ndim == 1 and xi_arr.shape == ez_arr.shape:
            xs = intruder_x + ez_arr * np.cos(xi_arr)
            ys = intruder_y + ez_arr * np.sin(xi_arr)
            points = list(zip(xs.tolist(), ys.tolist()))
    except Exception:
        points = []

    # Case B: DataFrame-like or list-of-(x,y)
    if not points:
        try:
            if hasattr(ez_points, "itertuples"):
                pts = []
                for row in ez_points.itertuples(index=False):
                    x = getattr(row, "x", None)
                    y = getattr(row, "y", None)
                    if x is None:
                        x = row[0]
                    if y is None:
                        y = row[1]
                    pts.append((float(x), float(y)))
                points = pts
            else:
                pts = []
                for p in ez_points:
                    pts.append((float(p[0]), float(p[1])))
                points = pts
        except Exception:
            points = []

    if not points:
        return []

    # single_circle approximation
    if mode == "single_circle":
        dists = [np.hypot(px - intruder_x, py - intruder_y) for px, py in points]
        max_r = max(dists) if dists else 0.0
        out_r = float(max_r + (obstacle_radius or 0.0))
        return [(float(intruder_x), float(intruder_y), out_r)]

    # hull_vertices mode
    if mode == "hull_vertices":
        pts_arr = np.asarray(points)
        if pts_arr.shape[0] <= 3:
            hull_pts = pts_arr
        else:
            pts_sorted = pts_arr[np.lexsort((pts_arr[:, 1], pts_arr[:, 0]))]

            def _half_hull(points_arr):
                hull = []
                for p in points_arr:
                    while len(hull) >= 2:
                        q1 = hull[-2]
                        q2 = hull[-1]
                        cross = (q2[0] - q1[0]) * (p[1] - q1[1]) - (q2[1] - q1[1]) * (p[0] - q1[0])
                        if cross <= 0:
                            hull.pop()
                        else:
                            break
                    hull.append(tuple(p))
                return hull

            lower = _half_hull(pts_sorted)
            upper = _half_hull(pts_sorted[::-1])
            hull = lower[:-1] + upper[:-1]
            hull_pts = np.asarray(hull)

        if hull_subsample and hull_pts.shape[0] > hull_subsample:
            hull_pts = hull_pts[::hull_subsample]
        r = float(obstacle_radius or 0.0)
        return [(float(px), float(py), r) for px, py in hull_pts]

    # default: per_point
    if subsample and subsample > 1:
        points = points[::subsample]

    r = float(obstacle_radius or 0.0)
    return [(float(px), float(py), r) for px, py in points]
