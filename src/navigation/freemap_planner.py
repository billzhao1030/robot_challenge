from __future__ import annotations

import math

import cv2
import numpy as np
from pxr import UsdGeom


class FreemapPlanner:
    def __init__(self, freemap_path: str, safety_margin_m: float):
        self.occupancy = np.load(freemap_path)
        self.x_coords = self.occupancy[0, 1:]
        self.y_coords = self.occupancy[1:, 0]
        self.raw_grid = self.occupancy[1:, 1:].astype(np.int8)
        self.grid_resolution = float(np.median(np.abs(np.diff(self.x_coords))))
        self.grid = self._inflate_obstacles(self.raw_grid, safety_margin_m)
        self.height, self.width = self.grid.shape

    def _inflate_obstacles(self, raw_grid: np.ndarray, safety_margin_m: float) -> np.ndarray:
        free_mask = raw_grid == 1
        blocked_mask = ~free_mask
        radius_cells = max(0, int(math.ceil(safety_margin_m / max(self.grid_resolution, 1.0e-6))))
        if radius_cells == 0:
            return free_mask.astype(np.int8)

        kernel_size = radius_cells * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        cv2.circle(kernel, (radius_cells, radius_cells), radius_cells, 1, thickness=-1)
        inflated_blocked = cv2.dilate(blocked_mask.astype(np.uint8), kernel, iterations=1) > 0
        safe_free_mask = free_mask & (~inflated_blocked)
        return safe_free_mask.astype(np.int8)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        idx_x = int(np.argmin(np.abs(self.x_coords - x)))
        idx_y = int(np.argmin(np.abs(self.y_coords - y)))
        return idx_x, idx_y

    def grid_to_world(self, idx_x: int, idx_y: int) -> tuple[float, float]:
        return float(self.x_coords[idx_x]), float(self.y_coords[idx_y])

    def is_free(self, idx_x: int, idx_y: int) -> bool:
        return 0 <= idx_x < self.width and 0 <= idx_y < self.height and self.grid[idx_y, idx_x] == 1

    def find_nearest_reachable(self, x: float, y: float, max_search_radius: int = 30) -> tuple[float, float, int] | None:
        from collections import deque

        idx_x, idx_y = self.world_to_grid(x, y)
        if self.is_free(idx_x, idx_y):
            return self.grid_to_world(idx_x, idx_y) + (0,)

        queue = deque([(idx_y, idx_x, 0)])
        visited = {(idx_y, idx_x)}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        min_dist_found = float("inf")
        reachable_candidates: list[tuple[int, int]] = []

        while queue:
            cy, cx, dist = queue.popleft()
            if dist > max_search_radius or dist >= min_dist_found:
                break

            next_dist = dist + 1
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < self.height and 0 <= nx < self.width) or (ny, nx) in visited:
                    continue
                visited.add((ny, nx))

                if self.grid[ny, nx] == 1:
                    if next_dist < min_dist_found:
                        min_dist_found = next_dist
                        reachable_candidates = []
                    if next_dist == min_dist_found:
                        reachable_candidates.append((ny, nx))

                if next_dist < min_dist_found:
                    queue.append((ny, nx, next_dist))

        if not reachable_candidates:
            return None

        best_candidate = None
        min_euclidean_dist_sq = float("inf")
        for ny, nx in reachable_candidates:
            candidate_x, candidate_y = self.grid_to_world(nx, ny)
            dist_sq = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
            if dist_sq < min_euclidean_dist_sq:
                min_euclidean_dist_sq = dist_sq
                best_candidate = (candidate_x, candidate_y, min_dist_found)
        return best_candidate


def heuristic(node: tuple[int, int], goal: tuple[int, int]) -> float:
    return math.hypot(goal[0] - node[0], goal[1] - node[1])


def freemap_segment_clear(
    planner: FreemapPlanner,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    samples: int = 40,
) -> bool:
    for alpha in np.linspace(0.0, 1.0, samples):
        point = (1.0 - alpha) * start_xy + alpha * end_xy
        idx_x, idx_y = planner.world_to_grid(float(point[0]), float(point[1]))
        if not planner.is_free(idx_x, idx_y):
            return False
    return True


def astar_on_freemap(
    planner: FreemapPlanner,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> np.ndarray:
    import heapq

    start_fix = planner.find_nearest_reachable(float(start_xy[0]), float(start_xy[1]))
    goal_fix = planner.find_nearest_reachable(float(goal_xy[0]), float(goal_xy[1]))
    if start_fix is None or goal_fix is None:
        raise RuntimeError(f"Could not find reachable start/goal around {start_xy.tolist()} -> {goal_xy.tolist()}.")

    start_px = planner.world_to_grid(start_fix[0], start_fix[1])
    goal_px = planner.world_to_grid(goal_fix[0], goal_fix[1])
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start_px, goal_px), start_px)]
    g_score = {start_px: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start_px: None}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_px:
            break

        for dx, dy, step_cost in neighbors:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or nx >= planner.width or ny < 0 or ny >= planner.height:
                continue
            if not planner.is_free(nx, ny):
                continue
            if dx != 0 and dy != 0:
                if not planner.is_free(current[0] + dx, current[1]) or not planner.is_free(current[0], current[1] + dy):
                    continue
            neighbor = (nx, ny)
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                parent[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal_px), neighbor))

    if goal_px not in parent:
        raise RuntimeError(f"No collision-free path found from {start_xy.tolist()} to {goal_xy.tolist()}.")

    pixel_path = []
    node: tuple[int, int] | None = goal_px
    while node is not None:
        pixel_path.append(node)
        node = parent[node]
    pixel_path.reverse()

    return np.asarray([planner.grid_to_world(ix, iy) for ix, iy in pixel_path], dtype=np.float32)


def smooth_path_world(planner: FreemapPlanner, world_path: np.ndarray) -> np.ndarray:
    if len(world_path) <= 2:
        return world_path

    smoothed = [world_path[0]]
    anchor_idx = 0
    probe_idx = 1

    while probe_idx < len(world_path):
        if freemap_segment_clear(planner, world_path[anchor_idx], world_path[probe_idx]):
            probe_idx += 1
            continue
        smoothed.append(world_path[probe_idx - 1])
        anchor_idx = probe_idx - 1

    smoothed.append(world_path[-1])
    return np.asarray(smoothed, dtype=np.float32)


def plan_waypoint_path(
    planner: FreemapPlanner,
    start_xyz: tuple[float, float, float],
    goals_xyz: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    current_xy = np.asarray(start_xyz[:2], dtype=np.float32)
    planned_waypoints: list[tuple[float, float, float]] = []

    for goal_x, goal_y, goal_z in goals_xyz:
        goal_xy = np.asarray([goal_x, goal_y], dtype=np.float32)
        raw_path = astar_on_freemap(planner=planner, start_xy=current_xy, goal_xy=goal_xy)
        smooth_path = smooth_path_world(planner=planner, world_path=raw_path)
        for point_xy in smooth_path[1:]:
            planned_waypoints.append((float(point_xy[0]), float(point_xy[1]), goal_z))
        current_xy = goal_xy

    return planned_waypoints


def add_path_debug_vis(stage, path_points_xyz: list[tuple[float, float, float]]) -> None:
    debug_path = "/World/PlanDebug"
    if stage.GetPrimAtPath(debug_path):
        stage.RemovePrim(debug_path)
    if len(path_points_xyz) < 2:
        return

    curve = UsdGeom.BasisCurves.Define(stage, debug_path)
    points = [(float(x), float(y), 0.04) for x, y, _ in path_points_xyz]
    curve.CreatePointsAttr(points)
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWidthsAttr([0.05] * len(points))
    curve.CreateDisplayColorPrimvar().Set([(0.1, 0.9, 0.2)])
