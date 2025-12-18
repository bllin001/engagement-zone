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
from matplotlib import animation

from utils.engament_zone_model import agent_safety, rho_values, wrap_angle_deg


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
        self.intruder_parameters = {
            'x': float(intruder_parameters.get('x', 0.0)),
            'y': float(intruder_parameters.get('y', 0.0)),
            'speed': float(intruder_speed) if intruder_speed is not None else 0.0,
        }
        self.mu = float(mu)
        self.R = float(R)
        self.r = float(r)
        self.agent_speed = float(agent_speed) if agent_speed is not None else 0.0
        self.lambda_weight = float(lambda_weight)
        self.edge_sample_points = int(edge_sample_points)
        self.edge_log_limit = int(edge_log_limit)
        self.logged_accepts = 0
        self.logged_rejects = 0

        xi_vals, ez_radii, _ = rho_values(self.mu, self.R, self.r)
        self._xi_lookup = np.asarray(xi_vals, dtype=float)
        self._rho_lookup = np.asarray(ez_radii, dtype=float)

    def plan(self):
        """
        Main planning loop for RRT*.
        """
        for _ in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rand_node)
            new_node = self.steer(nearest_node, rand_node)

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
            return Node(random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))
        return Node(self.goal.x, self.goal.y)

    def steer(self, from_node, to_node):
        """
        Steer from one node to another by a fixed step size.
        """
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(
            from_node.x + self.step_size * math.cos(theta),
            from_node.y + self.step_size * math.sin(theta),
        )
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
        return path[::-1]

    def get_nearest_node(self, rand_node):
        """
        Get the nearest node in the tree to the given random node.
        """
        distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y]) for node in self.node_list]
        return self.node_list[int(np.argmin(distances))]

    def evaluate_edge_cost(self, from_node, to_node, log_decision=False):
        """
        Evaluate whether the edge is valid under EZ constraints and compute its combined cost.
        """
        heading_theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        heading_deg = math.degrees(heading_theta)

        sample_params = np.linspace(0.0, 1.0, self.edge_sample_points)
        clearances = []

        intruder = self.intruder_parameters
        intruder_dict = {
            'x': intruder['x'],
            'y': intruder['y'],
            'speed': intruder.get('speed', 0.0),
        }

        for t in sample_params:
            sample_x = from_node.x + t * (to_node.x - from_node.x)
            sample_y = from_node.y + t * (to_node.y - from_node.y)
            dx = sample_x - intruder['x']
            dy = sample_y - intruder['y']
            distance = math.hypot(dx, dy)

            theta_los_deg = math.degrees(math.atan2(dy, dx))
            aspect_deg = wrap_angle_deg(heading_deg - theta_los_deg)
            aspect_rad = math.radians(aspect_deg)
            ez_radius = self._lookup_rho(aspect_rad)

            agent_dict = {
                'x': sample_x,
                'y': sample_y,
                'heading': heading_theta,
                'speed': self.agent_speed,
            }
            safe, _, _ = agent_safety(intruder_dict, agent_dict, self.mu, self.R, self.r)

            if (not safe) or (distance <= ez_radius):
                if log_decision and self.logged_rejects < self.edge_log_limit:
                    self.logged_rejects += 1
                    print(
                        f"[EZ-EDGE] Rejected edge ({from_node.x:.2f},{from_node.y:.2f}) -> "
                        f"({to_node.x:.2f},{to_node.y:.2f}) at sample ({sample_x:.2f},{sample_y:.2f}); "
                        f"d={distance:.3f}, rho={ez_radius:.3f}, ξ={aspect_deg:.1f}°"
                    )
                return np.inf

            clearance = distance - ez_radius
            clearances.append(clearance)

        if not clearances:
            return np.inf

        D_max = max(clearances)
        if D_max <= 0:
            return np.inf
        clearance_min = min(clearances)

        L = np.linalg.norm([to_node.x - from_node.x, to_node.y - from_node.y])
        L_max = max(self.step_size, 1e-6)
        C_distance = L / L_max
        C_distance = min(C_distance, 1.0)

        C_safety = clearance_min / D_max
        C_edge = self.lambda_weight * C_distance + (1.0 - self.lambda_weight) * C_safety

        if log_decision and self.logged_accepts < self.edge_log_limit:
            self.logged_accepts += 1
            print(
                f"[EZ-EDGE] Accepted edge ({from_node.x:.2f},{from_node.y:.2f}) -> "
                f"({to_node.x:.2f},{to_node.y:.2f}); C_edge={C_edge:.3f}, "
                f"C_dist={C_distance:.3f}, C_safe={C_safety:.3f}"
            )
        return C_edge

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
