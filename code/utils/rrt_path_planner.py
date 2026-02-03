"""
RRT* path planner focused on Engagement Zone (EZ) avoidance experiments.

Exposes:
- Node: simple container for coordinates/cost/parent.
- RRTStar: planner plus visualization helpers. The `obstacles` parameter now
  represents a single Engagement Zone modeled as one circular obstacle centered
  on the intruder.
"""

# Organized imports
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from utils.engament_zone_model import agent_safety, rho_values, wrap_angle_deg
from utils.wind_uncertainty_model import wind_model, ground_speed_uncertainty


# Node class
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.edge_cost = 0.0


# RRTStar class
class RRTStar:
    def __init__(
        self,
        start,
        goal,
        map_size,
        obstacles=None,
        step_size=1.0,
        max_iter=500,
        goal_radius=1.5,
        search_radius=2.0,
        *,
        intruder_parameters=None,
        mu=0.7,
        R=1.0,
        r=0.25,
        agent_speed=1.0,
        lambda_weight=0.5,
        edge_sample_points=15,
        edge_log_limit=5,
        wind_csv_path=None,
        wind_method="cons_expected",
        wind_direction_deg=0.0,
        t_fuel=0.0,
    ):
        """
        Initialize the RRT* planner.

        Parameters:
        - start: tuple (x, y) start coordinates
        - goal: tuple (x, y) goal coordinates
        - map_size: tuple (width, height) of the map
        - obstacles: list of tuples (x_center, y_center, radius) describing the Engagement Zone (EZ).
                     Only one EZ is expected; it will be treated as the no-go region for both
                     collision checks and visualization.
        - step_size: incremental step size for tree expansion
        - max_iter: maximum number of iterations to run the planner
        - goal_radius: radius around goal to consider goal reached
        - search_radius: radius to search for neighbors during rewiring
        """
        self.start = Node(float(start[0]), float(start[1]))
        self.goal = Node(float(goal[0]), float(goal[1]))
        self.map_size = tuple(map(float, map_size))
        # obstacles corresponds to the Engagement Zone (EZ)
        self.obstacles = obstacles if obstacles is not None else []
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.goal_region_radius = float(goal_radius)
        self.search_radius = float(search_radius)
        self.node_list = [self.start]
        self.path = None
        self.goal_reached = False

        if intruder_parameters is None:
            raise ValueError("intruder_parameters are required for EZ validation.")
        intruder_speed = intruder_parameters.get('speed', 0.0)
        intr_heading = intruder_parameters.get('heading', None)
        intr_heading_deg = float(intruder_parameters.get('heading_deg', 0.0))
        if intr_heading is None:
            intr_heading = math.radians(intr_heading_deg)

        # Preserve intruder motion params (heading/index_speed) if provided
        self.intruder_parameters = {
            'x': float(intruder_parameters.get('x', 0.0)),
            'y': float(intruder_parameters.get('y', 0.0)),
            'speed': float(intruder_speed) if intruder_speed is not None else 0.0,
            'heading': float(intr_heading),
            'heading_deg': float(intr_heading_deg),
            'index_speed': float(intruder_parameters.get('index_speed', 0.0)),
        }
        self.mu = float(mu)
        self.R = float(R)
        self.r = float(r)
        self.agent_speed = float(agent_speed) if agent_speed is not None else 0.0
        self.lambda_weight = float(lambda_weight)
        
        # --- Wind grid (loaded once) -----------------------------------------
        self.wind_csv_path = wind_csv_path
        self.wind_method = str(wind_method)
        self.wind_direction_deg = float(wind_direction_deg)
        self.wind_direction_rad = math.radians(self.wind_direction_deg)
        self.t_fuel = float(t_fuel)

        if wind_csv_path is not None:
            df = pd.read_csv(wind_csv_path)
            # index for O(1) lookup: (lon_idx, lat_idx)
            df["lon_idx"] = df["lon_idx"].astype(int)
            df["lat_idx"] = df["lat_idx"].astype(int)
            self.wind_df = df.set_index(["lon_idx", "lat_idx"]).sort_index()
        else:
            self.wind_df = None
        
        self.edge_sample_points = int(edge_sample_points)
        self.edge_log_limit = int(edge_log_limit)
        self.logged_accepts = 0
        self.logged_rejects = 0

        xi_vals, ez_radii, _ = rho_values(self.mu, self.R, self.r)
        self._xi_lookup = np.asarray(xi_vals, dtype=float)
        self._rho_lookup = np.asarray(ez_radii, dtype=float)
    
    def get_wind_at_cell(self, ix: int, iy: int):
        """
        Return wind regimes for grid cell (ix, iy) using the environment CSV.

        Output keys are normalized to:
        - cell_id
        - ws1, ws2
        - p1, p2
        - std1, std2 (optional, may be NaN)
        """
        if self.wind_df is None:
            return None

        try:
            row = self.wind_df.loc[(int(ix), int(iy))]
        except KeyError:
            return None

        cell_id = row.get("cell_id", np.nan)

        ws1 = float(row["wind_speed_1"])
        p1 = float(row["probability_1"])

        ws2 = row.get("wind_speed_2", 0.0)
        p2 = row.get("probability_2", 0.0)

        std1 = row.get("std_1", np.nan)
        std2 = row.get("std_2", np.nan)

        # Convert NaN -> 0 for unimodal rows
        ws2 = 0.0 if (ws2 is None or (isinstance(ws2, float) and math.isnan(ws2))) else float(ws2)
        p2  = 0.0 if (p2  is None or (isinstance(p2,  float) and math.isnan(p2 ))) else float(p2)

        # Normalize probabilities just in case
        psum = p1 + p2
        if psum > 0:
            p1, p2 = p1 / psum, p2 / psum
        else:
            p1, p2 = 1.0, 0.0

        return {
            "cell_id": cell_id,
            "ws1": ws1, "ws2": ws2,
            "p1": p1, "p2": p2,
            "std1": float(std1) if std1 == std1 else np.nan,
            "std2": float(std2) if std2 == std2 else np.nan,
        }

    def plan(self):
        """
        Main planning loop for RRT*.
        """
        for _ in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rand_node)
            new_node = self.steer(nearest_node, rand_node)

            if new_node is None:
                continue

            if not self.is_collision_free(new_node):
                continue

            edge_cost = self.evaluate_edge_cost(nearest_node, new_node, log_decision=True)
            if not np.isfinite(edge_cost):
                continue

            new_node.parent = nearest_node
            new_node.edge_cost = edge_cost
            new_node.cost = nearest_node.cost + edge_cost

            neighbors = self.find_neighbors(new_node)
            new_node = self.choose_parent(neighbors, new_node)
            self.node_list.append(new_node)
            self.rewire(new_node, neighbors)

            if self.reached_goal(new_node):
                self.path = self.generate_final_path(new_node)
                self.goal_reached = True
                break
        return self.path

    def get_random_node(self):
        """
        Get a random node within the map boundaries or return the goal node with
        a small probability.
        """
        if random.random() > 0.2:
            return Node(
                random.randint(0, int(self.map_size[0]) - 1), 
                random.randint(0, int(self.map_size[1]) - 1)
                )
        return Node(int(round(self.goal.x)), int(round(self.goal.y)))

    def steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y

        step = int(round(self.step_size))
        step = max(step, 1)

        sx = 0 if dx == 0 else (1 if dx > 0 else -1)
        sy = 0 if dy == 0 else (1 if dy > 0 else -1)

        new_x = from_node.x + sx * step
        new_y = from_node.y + sy * step

        # clamp a límites del mapa
        new_x = max(0, min(new_x, int(self.map_size[0]) - 1))
        new_y = max(0, min(new_y, int(self.map_size[1]) - 1))

        # evita nodo idéntico (por si ya está pegado al borde)
        if new_x == from_node.x and new_y == from_node.y:
            return None

        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        return new_node

    def is_collision_free(self, node):
        """
        Check if the path to the given node is collision-free with respect to
        the Engagement Zone (EZ).
        """
        for (ox, oy, size) in self.obstacles:
            if (node.x - ox) ** 2 + (node.y - oy) ** 2 <= size ** 2:
                return False
        return True

    def find_neighbors(self, new_node):
        """
        Find neighboring nodes within the search radius.
        """
        return [
            node
            for node in self.node_list
            if np.linalg.norm([node.x - new_node.x, node.y - new_node.y]) < self.search_radius
        ]

    def choose_parent(self, neighbors, new_node):
        """
        Choose the best parent for the new node from its neighbors given EZ-aware edge costs.
        """
        best_parent = new_node.parent
        best_cost = new_node.cost
        best_edge_cost = new_node.edge_cost

        for neighbor in neighbors:
            edge_cost = self.evaluate_edge_cost(neighbor, new_node)
            if not np.isfinite(edge_cost):
                continue
            candidate_cost = neighbor.cost + edge_cost
            if candidate_cost < best_cost:
                best_parent = neighbor
                best_cost = candidate_cost
                best_edge_cost = edge_cost

        new_node.parent = best_parent
        new_node.cost = best_cost
        new_node.edge_cost = best_edge_cost
        return new_node

    def rewire(self, new_node, neighbors):
        """
        Rewire the tree by updating the parents of the neighbors of the new node
        if the path through the new node is cheaper.
        """
        for neighbor in neighbors:
            edge_cost = self.evaluate_edge_cost(new_node, neighbor)
            if not np.isfinite(edge_cost):
                continue
            cost = new_node.cost + edge_cost
            if cost < neighbor.cost and self.is_collision_free(neighbor):
                neighbor.parent = new_node
                neighbor.cost = cost
                neighbor.edge_cost = edge_cost

    def reached_goal(self, node):
        """
        Check if the given node is within the goal region.
        """
        return np.linalg.norm([node.x - self.goal.x, node.y - self.goal.y]) < self.goal_region_radius

    def generate_final_path(self, goal_node):
        """
        Generate the final path from start to goal by backtracking from the goal node.
        """
        path = []
        node = goal_node
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        path = path[::-1]
        if path and self.reached_goal(goal_node):
            path[-1] = [self.goal.x, self.goal.y]
        return path

    def get_nearest_node(self, rand_node):
        """
        Get the nearest node in the tree to the given random node.
        """
        distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y]) for node in self.node_list]
        return self.node_list[int(np.argmin(distances))]

    def evaluate_edge_cost(self, from_node, to_node, log_decision=False):
        """
        Edge cost using wind uncertainty (per-cell) + heading-dependent EZ.
        Uses wind_model() to compute mu and R for the edge, then computes rho_values(mu, R, r).
        """
        # 1) Heading del edge (depende del movimiento en grid)
        heading_theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        heading_deg = math.degrees(heading_theta)

        # 2) Celda destino (x=lon_idx, y=lat_idx)
        ix = int(round(to_node.x))
        iy = int(round(to_node.y))
        ix = max(0, min(ix, int(self.map_size[0]) - 1))
        iy = max(0, min(iy, int(self.map_size[1]) - 1))

        wind = self.get_wind_at_cell(ix, iy)
        if wind is None:
            return np.inf

        # 3) Construir parámetros para wind_model (heading importa)
        agent_params = {
            "speed": float(self.agent_speed),
            "heading": float(heading_theta),  # radians
        }
        intr_heading = self.intruder_parameters.get("heading", None)
        if intr_heading is None:
            intr_heading = math.radians(float(self.intruder_parameters.get("heading_deg", 0.0)))

        intruder_params = {
            "speed": float(self.intruder_parameters.get("speed", 0.0)),
            "heading": float(intr_heading),
        }

        res = wind_model(
            agent_params,
            intruder_params,
            t_fuel=self.t_fuel,
            method=self.wind_method,
            wind_speed_1=wind["ws1"],
            wind_speed_2=wind["ws2"],
            prob_w1=wind["p1"],
            R_baseline=self.R,                 # baseline R de tu setup
            wind_direction=self.wind_direction_rad,
        )

        mu_edge = float(res["mu"])
        R_edge = float(res["R"])

        # 4) Recalcular lookup de rho para este edge (porque mu/R cambian)
        xi_vals, ez_radii, _ = rho_values(mu_edge, R_edge, self.r)
        xi_lookup = np.asarray(xi_vals, dtype=float)
        rho_lookup = np.asarray(ez_radii, dtype=float)

        def lookup_rho_dynamic(xi_rad):
            xi = ((xi_rad + math.pi) % (2.0 * math.pi)) - math.pi
            idx = int(np.argmin(np.abs(xi_lookup - xi)))
            return float(rho_lookup[idx])

        # 5) Sampleo del edge (en grid con step=1, puedes bajar edge_sample_points a 1 o 2)
        sample_params = np.linspace(0.0, 1.0, self.edge_sample_points)
        clearances = []

        intruder = self.intruder_parameters
        intruder_dict = {
            "x": float(intruder["x"]),
            "y": float(intruder["y"]),
            "speed": float(intruder.get("speed", 0.0)),
        }

        for t in sample_params:
            sample_x = from_node.x + t * (to_node.x - from_node.x)
            sample_y = from_node.y + t * (to_node.y - from_node.y)

            dx = sample_x - intruder_dict["x"]
            dy = sample_y - intruder_dict["y"]
            distance = math.hypot(dx, dy)

            theta_los_deg = math.degrees(math.atan2(dy, dx))
            aspect_deg = wrap_angle_deg(heading_deg - theta_los_deg)
            aspect_rad = math.radians(aspect_deg)
            ez_radius = lookup_rho_dynamic(aspect_rad)

            agent_dict = {
                "x": sample_x,
                "y": sample_y,
                "heading": heading_theta,
                "speed": float(self.agent_speed),
            }

            safe, _, _ = agent_safety(intruder_dict, agent_dict, mu_edge, R_edge, self.r)

            if (not safe) or (distance <= ez_radius):
                if log_decision and self.logged_rejects < self.edge_log_limit:
                    self.logged_rejects += 1
                    print(
                        f"[EZ+WIND] Reject ({from_node.x:.0f},{from_node.y:.0f})->({to_node.x:.0f},{to_node.y:.0f}) "
                        f"cell=({ix},{iy}) ws=({wind['ws1']:.2f},{wind['ws2']:.2f}) p=({wind['p1']:.2f},{wind['p2']:.2f}) "
                        f"mu={mu_edge:.3f} R={R_edge:.3f} d={distance:.3f} rho={ez_radius:.3f} xi={aspect_deg:.1f}"
                    )
                return np.inf

            clearances.append(distance - ez_radius)

        if not clearances:
            return np.inf

        D_max = max(clearances)
        if D_max <= 0:
            return np.inf

        clearance_min = min(clearances)

        # distancia del edge
        L = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        L_max = max(self.step_size, 1e-9)  # en grid es ~1
        C_distance = min(L / L_max, 1.0)

        C_safety = clearance_min / D_max
        C_edge = self.lambda_weight * C_distance + (1.0 - self.lambda_weight) * C_safety

        if log_decision and self.logged_accepts < self.edge_log_limit:
            self.logged_accepts += 1
            print(
                f"[EZ+WIND] Accept ({from_node.x:.0f},{from_node.y:.0f})->({to_node.x:.0f},{to_node.y:.0f}) "
                f"cell=({ix},{iy}) mu={mu_edge:.3f} R={R_edge:.3f} C_edge={C_edge:.3f}"
            )

        return C_edge
    
    def export_simulation_csvs(
        self,
        path,
        output_dir,
        *,
        dt=1.0,
        n_theta=6,
        theta_min=-math.pi,
        theta_max=math.pi,
        endpoint=False,
        run_id=None,
    ):
        """
        Export simulation outputs into three clean CSV files:
        1) agent_state.csv   : agent indices (lon_idx/lat_idx) + selected wind regimes per t
        2) intruder_state.csv: intruder indices + EZ parameters (mu, R, r) per t
        3) ez_boundary.csv   : EZ boundary points in index-space per t

        Assumption:
        - The planner operates directly in grid-index space:
            node.x == lon_idx, node.y == lat_idx
        - Therefore the exported data contains ONLY indices (no x/y world coords).

        Intruder motion:
        - If self.intruder_parameters provides 'index_speed' (> 0) and an optional
          'heading' (radians) or 'heading_deg' (degrees), the intruder position will
          advance each time step by:
              dx = index_speed * cos(heading) * dt
              dy = index_speed * sin(heading) * dt
        - Otherwise, intruder remains fixed at its initial x/y.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import math

        if path is None or len(path) == 0:
            raise ValueError("export_simulation_csvs: path is empty")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Intruder position (float in index-space for motion), plus speed/heading
        intr_x = float(self.intruder_parameters.get("x", 0.0))
        intr_y = float(self.intruder_parameters.get("y", 0.0))
        intr_speed = float(self.intruder_parameters.get("speed", 0.0))
        # Optional motion parameters
        heading_rad = self.intruder_parameters.get("heading", None)
        if heading_rad is None:
            heading_deg = float(self.intruder_parameters.get("heading_deg", 0.0))
            heading_rad = math.radians(heading_deg)
        index_speed = float(self.intruder_parameters.get("index_speed", 0.0))

        width = int(self.map_size[0])
        height = int(self.map_size[1])

        thetas = np.linspace(float(theta_min), float(theta_max), int(n_theta), endpoint=bool(endpoint))

        agent_rows = []
        intruder_rows = []
        boundary_rows = []
        interaction_rows = []

        for i, (lon_idx, lat_idx) in enumerate(path):
            lon_idx = int(round(lon_idx))
            lat_idx = int(round(lat_idx))

            # Clamp indices to map bounds
            lon_idx = max(0, min(lon_idx, width - 1))
            lat_idx = max(0, min(lat_idx, height - 1))

            # Heading in index space (computed from neighbor segment)
            if i < len(path) - 1:
                nlon, nlat = path[i + 1]
                heading = math.atan2(int(round(nlat)) - lat_idx, int(round(nlon)) - lon_idx)
            elif i > 0:
                plon, plat = path[i - 1]
                heading = math.atan2(lat_idx - int(round(plat)), lon_idx - int(round(plon)))
            else:
                heading = 0.0

            heading_deg = math.degrees(heading)
            t = float(i) * float(dt)

            # Determine current intruder cell (allow motion)
            intr_lon_int = max(0, min(int(round(intr_x)), width - 1))
            intr_lat_int = max(0, min(int(round(intr_y)), height - 1))

            # Read wind regimes for the current grid cell (agent) and intruder cell (possibly moving)
            wind_agent = self.get_wind_at_cell(lon_idx, lat_idx)
            wind_intr = self.get_wind_at_cell(intr_lon_int, intr_lat_int)
            
            agent_speed_eff = np.nan
            agent_speed_w1 = np.nan
            agent_speed_w2 = np.nan

            intruder_speed_eff = np.nan
            intruder_speed_w1 = np.nan
            intruder_speed_w2 = np.nan


           # Recompute mu, R, and wind-adjusted speeds for this step (agent-side wind)
            if wind_agent is not None:
                mu_R = wind_model(
                    {"speed": float(self.agent_speed), "heading": float(heading)},
                    {"speed": float(intr_speed), "heading": 0.0},
                    t_fuel=float(self.t_fuel),
                    method=str(self.wind_method),
                    wind_speed_1=float(wind_agent["ws1"]),
                    wind_speed_2=float(wind_agent["ws2"]),
                    prob_w1=float(wind_agent["p1"]),
                    R_baseline=float(self.R),
                    wind_direction=float(self.wind_direction_rad),
                )

                mu_t = float(mu_R["mu"])
                R_t = float(mu_R["R"])

                agent_g = mu_R["agent_ground_speeds"]

                agent_speed_eff = float(agent_g["expected"])
                agent_speed_w1  = float(agent_g["wind_1"])
                agent_speed_w2  = float(agent_g["wind_2"])

            else:
                mu_t = float(self.mu)
                R_t = float(self.R)

            # Intruder ground speed based on its own cell wind (keeps cell_id aligned with lon/lat)
            if wind_intr is not None:
                intruder_speed_eff, intruder_speed_w1, intruder_speed_w2 = ground_speed_uncertainty(
                    {"speed": float(intr_speed), "heading": 0.0},
                    wind_intr["ws1"],
                    wind_intr["ws2"],
                    wind_intr["p1"],
                    wind_direction=float(self.wind_direction_rad),
                )
            else:
                intruder_speed_eff = float(intr_speed)
                intruder_speed_w1 = float(intr_speed)
                intruder_speed_w2 = float(intr_speed)

            # --- Agent state (indices only)
            agent_rows.append({
                "t": t,
                "cell_id": int(wind_agent["cell_id"]) if wind_agent else np.nan,
                "lon_idx": lon_idx,
                "lat_idx": lat_idx,
                "agent_speed": float(self.agent_speed),
                "agent_ground_speed_expected": agent_speed_eff,
                "agent_ground_speed_w1": agent_speed_w1,
                "agent_ground_speed_w2": agent_speed_w2,
                "heading_deg": float(heading_deg),
                "wind_speed_1": float(wind_agent["ws1"]) if wind_agent else np.nan,
                "wind_speed_2": float(wind_agent["ws2"]) if wind_agent else np.nan,
                "probability_1": float(wind_agent["p1"]) if wind_agent else np.nan,
                "probability_2": float(wind_agent["p2"]) if wind_agent else np.nan,
            })

            # --- Intruder state (indices only) + EZ parameters
            intruder_rows.append({
                "t": t,
                "cell_id": int(wind_intr["cell_id"]) if wind_intr else np.nan,
                "lon_idx": intr_lon_int,
                "lat_idx": intr_lat_int,
                "intruder_speed": float(intr_speed),
                "intruder_ground_speed_expected": intruder_speed_eff if wind_intr else float(intr_speed),
                "intruder_ground_speed_w1": intruder_speed_w1 if wind_intr else float(intr_speed),
                "intruder_ground_speed_w2": intruder_speed_w2 if wind_intr else float(intr_speed),
                "mu": float(mu_t),
                "R": float(R_t),
                "r": float(self.r),
                "heading_deg": float(math.degrees(heading_rad)) if heading_rad is not None else 0.0,
                "index_speed": float(index_speed),
            })

            # --- EZ boundary sampled at N thetas, expressed in index-space
            xi_vals, rho_vals, _ = rho_values(float(mu_t), float(R_t), float(self.r))
            xi_arr = np.asarray(xi_vals, dtype=float)
            rho_arr = np.asarray(rho_vals, dtype=float)

            for j, theta in enumerate(thetas):
                idx = int(np.argmin(np.abs(xi_arr - float(theta))))
                rho = float(rho_arr[idx])

                b_lon = float(intr_x) + rho * math.cos(float(theta))
                b_lat = float(intr_y) + rho * math.sin(float(theta))

                boundary_rows.append({
                    "t": t,
                    "point_id": int(j),
                    "theta_rad": float(theta),
                    "rho": float(rho),
                    "boundary_lon_idx": float(b_lon),
                    "boundary_lat_idx": float(b_lat),
                })
            
            dx = lon_idx - intr_x
            dy = lat_idx - intr_y
            distance = math.sqrt(dx*dx + dy*dy)

            los_rad = math.atan2(dy, dx)  # intruder -> agent LOS

            xi_vals, rho_vals, _ = rho_values(mu_t, R_t, self.r)
            xi_arr = np.asarray(xi_vals, dtype=float)
            rho_arr = np.asarray(rho_vals, dtype=float)

            # find rho at LOS (closest theta sample)
            idx = int(np.argmin(np.abs(xi_arr - los_rad)))
            rho_at_los = float(rho_arr[idx])

            clearance = float(distance - rho_at_los)
            inside_EZ = bool(distance <= rho_at_los)

            interaction_rows.append({
                "t": t,
                "cell_id": int(wind_agent["cell_id"]) if wind_agent else np.nan,
                "agent_lon_idx": lon_idx,
                "agent_lat_idx": lat_idx,
                "intruder_lon_idx": intr_lon_int,
                "intruder_lat_idx": intr_lat_int,
                "distance": distance,
                "los_rad": los_rad,
                "rho_at_los": rho_at_los,
                "clearance": clearance,
                "inside_EZ": inside_EZ,
                "mu": mu_t,
                "R": R_t,
                "r": float(self.r),
            })

            # Advance intruder for next step if motion is enabled
            if index_speed != 0.0:
                intr_x = float(intr_x + index_speed * math.cos(heading_rad) * float(dt))
                intr_y = float(intr_y + index_speed * math.sin(heading_rad) * float(dt))
                # Clamp to map bounds
                intr_x = max(0.0, min(intr_x, float(width - 1)))
                intr_y = max(0.0, min(intr_y, float(height - 1)))

        pd.DataFrame(agent_rows).to_csv(out_dir / "agent_state.csv", index=False)
        pd.DataFrame(intruder_rows).to_csv(out_dir / "intruder_state.csv", index=False)
        pd.DataFrame(boundary_rows).to_csv(out_dir / "ez_boundary.csv", index=False)
        pd.DataFrame(interaction_rows).to_csv(out_dir / "interaction_state.csv", index=False)
    
    def _lookup_rho(self, xi_rad):
        """
        Get the EZ radius for a given aspect angle using nearest-neighbor lookup on rho_values output.
        """
        xi = ((xi_rad + math.pi) % (2.0 * math.pi)) - math.pi
        idx = int(np.argmin(np.abs(self._xi_lookup - xi)))
        return float(self._rho_lookup[idx])

    def plot(self, intruder=None, R=None, r=None, origin_shift=None, figsize=(12, 10), ax=None):
        """
        Draw the RRT* map and optionally annotate an intruder, its ranges, and an EZ boundary.

        Parameters:
        - intruder: tuple (x, y) in the same coordinate frame as the planner (shifted if planner inputs were shifted).
        - R: pursuer maximum range (drawn as dashed circle around intruder if provided).
        - r: neutralization radius (drawn as dotted circle around intruder if provided).
        - origin_shift: (shift_x, shift_y) pair used to move the visualization back to world coordinates.
        - figsize: matplotlib figure size tuple.
        - ax: optional matplotlib axis to plot on.
        - animation_mode: boolean flag to enable animation saving as .gif.

        Returns: (fig, ax)
        """
        # --- Academic Style RRT* Plot ---
        shift_x, shift_y = (origin_shift if origin_shift is not None else (0.0, 0.0))
        def to_world(x, y):
            return x - shift_x, y - shift_y

        plane_shape = np.array([
            [0.0, 1.2],
            [0.25, 0.45],
            [0.95, 0.25],
            [0.95, -0.25],
            [0.25, -0.45],
            [0.3, -0.8],
            [0.55, -1.1],
            [0.55, -1.35],
            [0.0, -1.05],
            [-0.55, -1.35],
            [-0.55, -1.1],
            [-0.3, -0.8],
            [-0.25, -0.45],
            [-0.95, -0.25],
            [-0.95, 0.25],
            [-0.25, 0.45],
        ])
        def add_plane(center, size, color, angle_deg=0, edgecolor='white', zorder=9, label=None):
            theta = np.deg2rad(angle_deg)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
            coords = (plane_shape @ rot.T) * size + np.array(center)
            patch = Polygon(
                coords,
                closed=True,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=1.4,
                zorder=zorder,
            )
            patch.set_path_effects([PathEffects.withStroke(linewidth=1.7, foreground='white')])
            if label:
                patch.set_label(label)
            ax.add_patch(patch)
            plane_extent = size * 1.4
            add_extent(center[0] + plane_extent, center[1] + plane_extent)
            add_extent(center[0] - plane_extent, center[1] - plane_extent)
            return patch

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # For axis extents
        world_points_x = []
        world_points_y = []
        def add_extent(x, y):
            world_points_x.append(float(x))
            world_points_y.append(float(y))

        # Collect points for extent
        for node in self.node_list + [self.start, self.goal]:
            wx, wy = to_world(node.x, node.y)
            add_extent(wx, wy)

        # Subtle grid and neutral background
        ax.set_facecolor("#f7f7fa")
        ax.grid(True, which='both', linestyle=':', color='#b4b9c9', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_aspect('equal', adjustable='box')

        # Plot tree edges in light steel blue, subtle alpha
        for node in self.node_list:
            if node.parent:
                x0, y0 = to_world(node.x, node.y)
                x1, y1 = to_world(node.parent.x, node.parent.y)
                ax.plot([x0, x1], [y0, y1], color='#7899c8', linewidth=0.85, alpha=0.35, zorder=1)

        # Engagement Zone (EZ) as translucent crimson circle with darker edge
        if self.obstacles:
            ox, oy, size = self.obstacles[0]
            wx, wy = to_world(ox, oy)
            ez_circle = plt.Circle(
                (wx, wy), size,
                facecolor='#d32c3a', edgecolor='#7a0010',
                linewidth=2.6, alpha=0.18,
                label='EZ (boundary)', zorder=3
            )
            ax.add_patch(ez_circle)
            ez_edge = plt.Circle(
                (wx, wy), size,
                facecolor='none', edgecolor='#7a0010',
                linewidth=2.2, alpha=0.55, zorder=3
            )
            ax.add_patch(ez_edge)
            add_extent(wx + size, wy)
            add_extent(wx - size, wy)
            add_extent(wx, wy + size)
            add_extent(wx, wy - size)
            add_extent(wx, wy)

        # Start: royal blue star
        start_world = to_world(self.start.x, self.start.y)
        ax.scatter(
            [start_world[0]], [start_world[1]],
            s=180, marker='*', color='#295ad2',
            edgecolor='white', linewidth=1.7,
            zorder=7, label='Start'
        )
        add_extent(start_world[0], start_world[1])

        # --- Agent plane marker (same aircraft shape as intruder, oriented toward goal) ---
        dx = self.goal.x - self.start.x
        dy = self.goal.y - self.start.y
        agent_angle = np.degrees(np.arctan2(dy, dx))
        add_plane(
            (start_world[0], start_world[1]),
            size=0.45,
            color='#295ad2',
            angle_deg=agent_angle,
            edgecolor='black',
            zorder=9,
            label='Agent'
        )

        # Goal: teal circle
        goal_world = to_world(self.goal.x, self.goal.y)
        ax.scatter(
            [goal_world[0]], [goal_world[1]],
            s=95, marker='o', color='#1baba5',
            edgecolor='white', linewidth=1.5,
            zorder=7, label='Goal'
        )
        add_extent(goal_world[0], goal_world[1])

        # Dashed blue guidance line from start to goal (for visualization)
        ax.plot(
            [start_world[0], goal_world[0]],
            [start_world[1], goal_world[1]],
            color='#295ad2',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            zorder=6
        )

        # Intruder: black polygon (plane shape)

        if intruder is not None:
            ix, iy = to_world(intruder[0], intruder[1])
            add_plane((ix, iy), size=0.55, color='#181818', angle_deg=205, edgecolor='#292929', zorder=10, label='Intruder')
            add_extent(ix, iy)
            ax.annotate(
                f"Intruder ({ix:.1f}, {iy:.1f})",
                xy=(ix, iy),
                textcoords="offset points",
                xytext=(20, -20),
                ha='left', fontsize=11, color='#222', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.5, alpha=0.8),
                clip_on=True,
            )
        if r is not None and r != 0:
            circr = plt.Circle(
                (ix, iy), r,
                color='#d67f2f', fill=False,
                linestyle=':', linewidth=2.1, alpha=0.68,
                label=f'r = {r}', zorder=5
            )
            ax.add_patch(circr)
            add_extent(ix + r, iy)
            add_extent(ix - r, iy)
            add_extent(ix, iy + r)
            add_extent(ix, iy - r)

        # Path: navy, slightly thicker
        if self.path:
            try:
                transformed = [to_world(px, py) for (px, py) in self.path]
                if transformed:
                    if np.hypot(transformed[0][0] - start_world[0], transformed[0][1] - start_world[1]) > 1e-6:
                        transformed.insert(0, start_world)
                    if np.hypot(transformed[-1][0] - goal_world[0], transformed[-1][1] - goal_world[1]) > 1e-6:
                        transformed.append(goal_world)
                xs_path, ys_path = zip(*transformed)
                for px, py in transformed:
                    add_extent(px, py)
                ax.plot(xs_path, ys_path, color='#12305b', linewidth=3.2, label='Final Path', zorder=12)
            except Exception as e:
                print(f"Error visualizing path: {e}")

        # Annotations for start/goal
        ax.annotate(
            f"Start ({start_world[0]:.1f}, {start_world[1]:.1f})",
            xy=start_world,
            textcoords="offset points",
            xytext=(10, 15),
            color='#295ad2',
            fontsize=11,
            fontweight='bold',
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#b4b9c9', lw=0.5, alpha=0.85),
            clip_on=True,
        )
        ax.annotate(
            f"Goal ({goal_world[0]:.1f}, {goal_world[1]:.1f})",
            xy=goal_world,
            textcoords="offset points",
            xytext=(-60, -25),
            color='#1baba5',
            fontsize=11,
            fontweight='bold',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#b4b9c9', lw=0.5, alpha=0.85),
            clip_on=True,
        )

        # Legend with white background and gray border
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {label: handle for handle, label in zip(handles, labels)}
        legend_order = ['Start', 'Goal', 'Final Path', 'EZ (boundary)', 'Intruder']
        if r is not None:
            legend_order.append(f'r = {r}')
        ordered_labels = [lab for lab in legend_order if lab in label_to_handle]
        ordered_handles = [label_to_handle[lab] for lab in ordered_labels]
        if ordered_handles:
            legend = ax.legend(
                ordered_handles,
                ordered_labels,
                loc='upper left',
                frameon=True,
                framealpha=1.0,
                facecolor='white',
                edgecolor='#b4b9c9',
                fontsize=12,
                handlelength=1.8,
                borderpad=0.8,
                columnspacing=1.1,
                labelspacing=0.8,
                fancybox=True
            )
            legend.get_frame().set_linewidth(1.1)
            legend.get_frame().set_edgecolor('#b4b9c9')

        # Adjust axis limits for path and all features
        if world_points_x and world_points_y:
            min_x, max_x = min(world_points_x), max(world_points_x)
            min_y, max_y = min(world_points_y), max(world_points_y)
            span_x = max_x - min_x
            span_y = max_y - min_y
            span = max(span_x, span_y, 1.0)
            pad = max(span * 0.19, 2.0)
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            half = span / 2.0 + pad
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

        # Axis labels and title with larger fonts
        ax.set_xlabel("X (units)", fontsize=15, fontweight='bold', color='#444444', labelpad=8)
        ax.set_ylabel("Y (units)", fontsize=15, fontweight='bold', color='#444444', labelpad=8)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.tick_params(axis='both', labelsize=13, colors='#555555', length=5, width=1.1)
        ax.set_title("RRT* Path Planning", fontsize=18, fontweight='bold', color='#2b2c36', pad=14)

        # Adjust figure margins and layout for better label visibility and export quality
        fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.16)
        try:
            plt.tight_layout()
        except Exception:
            pass

        # For publication-quality export, use:
        # plt.savefig("filename.png", bbox_inches='tight', dpi=300)
        return fig, ax
