from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from isaaclab.app import AppLauncher

# -------------------------
# 1) Parse args + launch app
# -------------------------
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--waypoints",
    type=float,
    nargs="+",
    default=[
        -1.5,
        1.6,
        0.8,
        0.0,
        0.0,
        0.8,
        0.0,
        1.0,
        0.8,
    ],
    help="Waypoint list as flat xyz xyz xyz...",
)
parser.add_argument("--move-speed", type=float, default=0.35, help="Root translation speed in m/s.")
parser.add_argument("--turn-speed-deg", type=float, default=90.0, help="Yaw rotation speed in deg/s.")
parser.add_argument(
    "--position-tolerance",
    type=float,
    default=0.05,
    help="Waypoint reach tolerance in meters.",
)
parser.add_argument(
    "--yaw-tolerance-deg",
    type=float,
    default=5.0,
    help="Allowed yaw error before starting translation.",
)
parser.add_argument("--map-cell-size", type=float, default=0.05, help="Occupancy map resolution in meters.")
parser.add_argument(
    "--map-margin",
    type=float,
    default=0.75,
    help="Extra planning margin added around the house bounds in meters.",
)
parser.add_argument(
    "--path-clearance",
    type=float,
    default=0.25,
    help="Obstacle inflation radius for planning in meters.",
)
parser.add_argument(
    "--plan-debug-vis",
    action="store_true",
    default=True,
    help="Draw occupancy map and planned path into the stage.",
)
parser.add_argument(
    "--freemap-path",
    type=str,
    default="/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/freemap.npy",
    help="Path to the precomputed occupancy freemap .npy file.",
)
parser.add_argument(
    "--walk-policy-path",
    type=str,
    default="",
    help="TorchScript locomotion policy path for G1. If empty, falls back to root-motion control.",
)
parser.add_argument(
    "--policy-decimation",
    type=int,
    default=4,
    help="Number of physics steps per low-level locomotion policy inference.",
)
parser.add_argument(
    "--policy-action-scale",
    type=float,
    default=0.5,
    help="Scale applied to locomotion policy outputs before adding the default joint pose.",
)
parser.add_argument(
    "--walk-speed",
    type=float,
    default=0.5,
    help="Target forward walking speed in m/s when using the locomotion policy.",
)
parser.add_argument(
    "--walk-yaw-rate",
    type=float,
    default=1.0,
    help="Maximum yaw rate command in rad/s for locomotion-policy walking.",
)
parser.add_argument(
    "--walk-heading-gain",
    type=float,
    default=1.8,
    help="Heading controller gain for locomotion-policy walking.",
)
parser.add_argument(
    "--walk-heading-stop-deg",
    type=float,
    default=35.0,
    help="Stop forward motion when heading error exceeds this angle in walking mode.",
)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -------------------------
# 2) Import Isaac Lab AFTER launching
# -------------------------
import numpy as np
import torch
import cv2

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab_assets import G1_CFG
from pxr import UsdGeom

from camera.floating_camera import FloatingCamera


@configclass
class HomeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    g1: ArticulationCfg = G1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 暂时不在房间里放人，先保留原始圆柱体配置但注释掉，不删除。
    #
    # human_0 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Humans/Human_0",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.18,
    #         height=1.75,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         activate_contact_sensors=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 0.0, 0.875]),
    # )
    #
    # human_1 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Humans/Human_1",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.17,
    #         height=1.62,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         activate_contact_sensors=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 1.0, 0.81]),
    # )
    #
    # human_2 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Humans/Human_2",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.14,
    #         height=1.10,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         activate_contact_sensors=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.5, 0.55]),
    # )


def setup_camera():
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=[-2.022355307502785, 1.617730767355105, 0.0],
        start_orientation=[0.687852177796288, 0.0, 0.0, -0.7258507983745032],
        camera_height=1.1,
    )
    camera.init_manual()
    camera.reset()
    return camera


def load_house_usd(env_index: int, usd_path: str, usd_root_prim: str) -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    house_prim_path = f"/World/envs/env_{env_index}/House"
    house_prim = stage.DefinePrim(house_prim_path, "Xform")
    house_prim.GetReferences().AddReference(usd_path, usd_root_prim)
    print(f"[OK] House referenced at {house_prim_path} from {usd_path} ({usd_root_prim})")


def parse_waypoints(raw_values: Sequence[float]) -> list[tuple[float, float, float]]:
    if len(raw_values) == 0 or len(raw_values) % 3 != 0:
        raise ValueError("--waypoints must contain xyz triplets.")
    return [
        (raw_values[i], raw_values[i + 1], raw_values[i + 2])
        for i in range(0, len(raw_values), 3)
    ]


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_prim_bounds(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(prim_path)
    world_bbox = bbox_cache.ComputeWorldBound(prim)
    aligned = world_bbox.ComputeAlignedBox()
    min_pt = np.array(aligned.GetMin(), dtype=np.float32)
    max_pt = np.array(aligned.GetMax(), dtype=np.float32)
    return min_pt, max_pt


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
        # Only value 1 is reachable. Values 0/2 are blocked.
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
            # For diagonal motion, both side-adjacent cells must also be free.
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

    world_path = np.asarray([planner.grid_to_world(ix, iy) for ix, iy in pixel_path], dtype=np.float32)
    return world_path


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
        raw_path = astar_on_freemap(
            planner=planner,
            start_xy=current_xy,
            goal_xy=goal_xy,
        )
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


class WaypointController:
    def __init__(
        self,
        robot,
        waypoints_xyz: list[tuple[float, float, float]],
        move_speed: float,
        turn_speed_rad: float,
        position_tolerance: float,
        yaw_tolerance_rad: float,
    ) -> None:
        self.robot = robot
        self.device = robot.device
        self.waypoints = torch.tensor(waypoints_xyz, dtype=torch.float32, device=self.device)
        self.move_speed = move_speed
        self.turn_speed_rad = turn_speed_rad
        self.position_tolerance = position_tolerance
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.current_index = 0
        self.finished = len(waypoints_xyz) == 0
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.zero_root_velocity = torch.zeros((self.robot.num_instances, 6), device=self.device)

    def initialize(self) -> None:
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.write_data_to_sim()
        self.robot.reset()

    def update(self, dt: float) -> None:
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.write_data_to_sim()
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)

        if self.finished:
            return

        current_pose = self.robot.data.root_pose_w.clone()
        current_pos = current_pose[:, :3]
        current_quat = current_pose[:, 3:7]
        target_pos = self.waypoints[self.current_index].view(1, 3)

        delta = target_pos - current_pos
        planar_delta = delta[:, :2]
        planar_distance = torch.linalg.norm(planar_delta, dim=-1)
        distance_3d = torch.linalg.norm(delta, dim=-1)

        _, _, current_yaw = math_utils.euler_xyz_from_quat(current_quat)
        desired_yaw = torch.where(
            planar_distance > 1.0e-6,
            torch.atan2(planar_delta[:, 1], planar_delta[:, 0]),
            current_yaw,
        )
        yaw_error = wrap_to_pi(desired_yaw - current_yaw)

        if torch.all(distance_3d <= self.position_tolerance):
            snapped_pose = current_pose.clone()
            snapped_pose[:, :3] = target_pos
            self.robot.write_root_pose_to_sim(snapped_pose)
            print(f"[Waypoint] reached #{self.current_index}: {target_pos[0].tolist()}")
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.finished = True
                print("[Waypoint] all waypoints completed.")
            return

        if torch.any((planar_distance > self.position_tolerance) & (torch.abs(yaw_error) > self.yaw_tolerance_rad)):
            yaw_step = torch.clamp(yaw_error, min=-self.turn_speed_rad * dt, max=self.turn_speed_rad * dt)
            next_yaw = current_yaw + yaw_step
            next_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(next_yaw),
                torch.zeros_like(next_yaw),
                next_yaw,
            )
            next_pose = current_pose.clone()
            next_pose[:, 3:7] = next_quat
            self.robot.write_root_pose_to_sim(next_pose)
            return

        step_length = min(self.move_speed * dt, float(distance_3d.max().item()))
        step_scale = step_length / max(float(distance_3d.max().item()), 1.0e-6)
        next_pos = current_pos + delta * step_scale
        next_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(desired_yaw),
            torch.zeros_like(desired_yaw),
            desired_yaw,
        )
        next_pose = current_pose.clone()
        next_pose[:, :3] = next_pos
        next_pose[:, 3:7] = next_quat
        self.robot.write_root_pose_to_sim(next_pose)


class PolicyWalkerController:
    def __init__(
        self,
        robot,
        waypoints_xyz: list[tuple[float, float, float]],
        position_tolerance: float,
        policy_path: str,
        low_level_decimation: int,
        policy_action_scale: float,
        walk_speed: float,
        max_yaw_rate: float,
        heading_gain: float,
        heading_stop_rad: float,
    ) -> None:
        self.robot = robot
        self.device = robot.device
        self.waypoints = torch.tensor(waypoints_xyz, dtype=torch.float32, device=self.device)
        self.position_tolerance = position_tolerance
        self.low_level_decimation = max(1, int(low_level_decimation))
        self.policy_action_scale = policy_action_scale
        self.walk_speed = walk_speed
        self.max_yaw_rate = max_yaw_rate
        self.heading_gain = heading_gain
        self.heading_stop_rad = heading_stop_rad
        self.current_index = 0
        self.finished = len(waypoints_xyz) == 0
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.last_actions = torch.zeros_like(self.default_joint_pos)
        self.zero_root_velocity = torch.zeros((self.robot.num_instances, 6), device=self.device)
        self._step_counter = 0

        if not check_file_path(policy_path):
            raise FileNotFoundError(
                f"Locomotion policy file '{policy_path}' does not exist or is not reachable from Isaac Sim."
            )
        self.policy = torch.jit.load(read_file(policy_path), map_location=self.device).eval()

    def initialize(self) -> None:
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.write_data_to_sim()
        self.robot.reset()

    def _advance_waypoint_if_reached(self) -> None:
        if self.finished:
            return

        current_pos = self.robot.data.root_pos_w.clone()
        target_pos = self.waypoints[self.current_index].view(1, 3)
        planar_error = torch.linalg.norm(target_pos[:, :2] - current_pos[:, :2], dim=-1)
        if torch.all(planar_error <= self.position_tolerance):
            print(f"[Walk] reached #{self.current_index}: {target_pos[0].tolist()}")
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.finished = True
                print("[Walk] all waypoints completed.")

    def _compute_velocity_command(self) -> torch.Tensor:
        velocity_command = torch.zeros((self.robot.num_instances, 3), dtype=torch.float32, device=self.device)
        if self.finished:
            return velocity_command

        current_pos = self.robot.data.root_pos_w.clone()
        current_quat = self.robot.data.root_quat_w.clone()
        target_pos = self.waypoints[self.current_index].view(1, 3)
        planar_delta = target_pos[:, :2] - current_pos[:, :2]
        planar_distance = torch.linalg.norm(planar_delta, dim=-1)

        _, _, current_yaw = math_utils.euler_xyz_from_quat(current_quat)
        desired_yaw = torch.atan2(planar_delta[:, 1], planar_delta[:, 0])
        yaw_error = wrap_to_pi(desired_yaw - current_yaw)

        forward_speed = torch.minimum(torch.full_like(planar_distance, self.walk_speed), planar_distance)
        if self.heading_stop_rad > 0.0:
            forward_speed = torch.where(
                torch.abs(yaw_error) > self.heading_stop_rad,
                torch.zeros_like(forward_speed),
                forward_speed * torch.clamp(torch.cos(yaw_error), min=0.0),
            )
        else:
            forward_speed = forward_speed * torch.clamp(torch.cos(yaw_error), min=0.0)

        yaw_rate = torch.clamp(self.heading_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)
        yaw_rate = torch.where(
            planar_distance <= self.position_tolerance,
            torch.zeros_like(yaw_rate),
            yaw_rate,
        )

        velocity_command[:, 0] = forward_speed
        velocity_command[:, 1] = 0.0
        velocity_command[:, 2] = yaw_rate
        return velocity_command

    def _compute_low_level_observation(self, velocity_command: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                velocity_command,
                self.robot.data.joint_pos - self.default_joint_pos,
                self.robot.data.joint_vel - self.default_joint_vel,
                self.last_actions,
            ),
            dim=-1,
        )

    def update(self, dt: float) -> None:
        del dt
        self._advance_waypoint_if_reached()
        velocity_command = self._compute_velocity_command()

        if self._step_counter % self.low_level_decimation == 0:
            with torch.inference_mode():
                self.last_actions = self.policy(self._compute_low_level_observation(velocity_command))

        joint_targets = self.default_joint_pos + self.policy_action_scale * self.last_actions
        self.robot.set_joint_position_target(joint_targets)
        self.robot.write_data_to_sim()
        self._step_counter += 1


def main():
    sim_dt = 0.005 if args.walk_policy_path else 0.01
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[-2.022355307502785, 1.617730767355105, 1.2],
        target=[0.0, 0.0, 0.0],
    )

    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Humans")

    usd_path = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
    load_house_usd(env_index=0, usd_path=usd_path, usd_root_prim="/Root")

    camera = None if args.headless else setup_camera()

    scene = InteractiveScene(HomeSceneCfg(num_envs=1, env_spacing=2.5))
    robot = scene["g1"]

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    initial_goals = parse_waypoints(args.waypoints)
    default_root_state = robot.data.default_root_state[0].clone()
    start_xyz = tuple(float(v) for v in default_root_state[:3].tolist())
    planner = FreemapPlanner(args.freemap_path, safety_margin_m=args.path_clearance)
    planned_waypoints = plan_waypoint_path(
        planner=planner,
        start_xyz=start_xyz,
        goals_xyz=initial_goals,
    )

    if args.plan_debug_vis:
        add_path_debug_vis(stage, [start_xyz] + planned_waypoints)

    use_walk_policy = bool(args.walk_policy_path)
    if use_walk_policy:
        controller = PolicyWalkerController(
            robot=robot,
            waypoints_xyz=planned_waypoints,
            position_tolerance=args.position_tolerance,
            policy_path=args.walk_policy_path,
            low_level_decimation=args.policy_decimation,
            policy_action_scale=args.policy_action_scale,
            walk_speed=args.walk_speed,
            max_yaw_rate=args.walk_yaw_rate,
            heading_gain=args.walk_heading_gain,
            heading_stop_rad=math.radians(args.walk_heading_stop_deg),
        )
    else:
        controller = WaypointController(
            robot=robot,
            waypoints_xyz=planned_waypoints,
            move_speed=args.move_speed,
            turn_speed_rad=math.radians(args.turn_speed_deg),
            position_tolerance=args.position_tolerance,
            yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
        )
    controller.initialize()

    print("[OK] Scene ready: house + G1.")
    print(f"[Waypoint] raw goals: {initial_goals}")
    print(f"[Waypoint] freemap resolution: {planner.grid_resolution:.4f} m, safety margin: {args.path_clearance:.3f} m")
    print(f"[Waypoint] planned path: {controller.waypoints.tolist()}")
    if use_walk_policy:
        print(f"[Walk] locomotion policy enabled: {args.walk_policy_path}")
        print(
            f"[Walk] sim dt: {sim_dt:.4f} s, policy decimation: {args.policy_decimation}, walk speed: {args.walk_speed:.2f} m/s"
        )
    else:
        print("[Walk] locomotion policy not provided. Falling back to root-motion controller.")

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        controller.update(dt)
        sim.step(render=True)
        if camera is not None:
            camera.run(dt)
        scene.update(dt)
        if controller.finished:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
