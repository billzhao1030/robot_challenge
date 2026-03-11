from __future__ import annotations

import math

from isaaclab.app import AppLauncher
import torch

from app_args import parse_main_args

# -------------------------
# 1) Parse args + launch app
# -------------------------
args = parse_main_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -------------------------
# 2) Import Isaac Lab AFTER launching
# -------------------------
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.physx
import omni.usd
from isaaclab.scene import InteractiveScene
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from camera.floating_camera import FloatingCamera
from world.scene_cfg import HomeSceneCfg


NAVIGATION_HEIGHT = 0.74
ROBOT_GOAL: tuple[float, float, float] = (-15.0, 0.0, NAVIGATION_HEIGHT)

OBSTACLE_PRIM_PATH = "/World/ObstacleCylinder"
OBSTACLE_XY: tuple[float, float] = (-10.0, 0.0)
OBSTACLE_BASE_Z = 0.0
OBSTACLE_RADIUS = 0.55
OBSTACLE_HEIGHT = 2.1

ROBOT_PRIM_PATH = "/World/envs/env_0/Robot"
RAYCAST_GUARD_ENABLED = True
RAYCAST_HEIGHTS = (0.12, 0.35, 0.75, 1.1)
RAYCAST_START_OFFSET = 0.35
RAYCAST_LOOKAHEAD = 1.0

STATUS_PRINT_EVERY_STEPS = 60
# Set to 0 to let the robot wait indefinitely while the path is blocked.
MAX_RUNTIME_STEPS = 0


def get_obstacle_center_xyz() -> tuple[float, float, float]:
    return (
        OBSTACLE_XY[0],
        OBSTACLE_XY[1],
        OBSTACLE_BASE_Z + 0.5 * OBSTACLE_HEIGHT,
    )


class RaycastGuardController:
    """Pose-stepping controller that pauses on forward ray hits."""

    def __init__(
        self,
        robot,
        goal_xyz: tuple[float, float, float],
        move_speed: float,
        turn_speed_rad: float,
        position_tolerance: float,
        yaw_tolerance_rad: float,
        use_raycast_guard: bool,
    ) -> None:
        self.robot = robot
        self.goal_xyz = goal_xyz
        self.move_speed = move_speed
        self.turn_speed_rad = turn_speed_rad
        self.position_tolerance = position_tolerance
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.use_raycast_guard = use_raycast_guard
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.device = self.default_joint_pos.device
        self.zero_root_velocity = torch.zeros((self.robot.num_instances, 6), device=self.device)
        self.finished = False
        self.blocked = False
        self._reported_hit_path: str | None = None

    def initialize(self) -> None:
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.write_data_to_sim()
        self.robot.reset()

    def get_xyz(self) -> tuple[float, float, float]:
        root_pos = self.robot.data.root_pos_w[0]
        return tuple(float(v) for v in root_pos.tolist())

    def _raycast_forward(self, x: float, y: float, dir_x: float, dir_y: float, max_dist: float) -> tuple[float, str] | None:
        if max_dist <= 0.0:
            return None

        origin_x = x + dir_x * RAYCAST_START_OFFSET
        origin_y = y + dir_y * RAYCAST_START_OFFSET
        query = omni.physx.get_physx_scene_query_interface()
        closest_hit: tuple[float, str] | None = None

        for ray_height in RAYCAST_HEIGHTS:
            origin = Gf.Vec3f(origin_x, origin_y, ray_height)
            direction = Gf.Vec3f(dir_x, dir_y, 0.0)
            hit = query.raycast_closest(origin, direction, max_dist)
            if not hit["hit"]:
                continue

            hit_path = str(hit.get("collision") or hit.get("rigidBody") or "")
            if hit_path == "":
                continue

            hit_distance = float(hit["distance"])
            if closest_hit is None or hit_distance < closest_hit[0]:
                closest_hit = (hit_distance, hit_path)

        return closest_hit

    def update(self, dt: float) -> None:
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.write_data_to_sim()
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)

        if self.finished:
            return

        current_pose = self.robot.data.root_pose_w.clone()
        current_pos = current_pose[:, :3]
        current_quat = current_pose[:, 3:7]

        target_pos = current_pos.clone()
        target_pos[:, 0] = self.goal_xyz[0]
        target_pos[:, 1] = self.goal_xyz[1]

        delta = target_pos - current_pos
        planar_delta = delta[:, :2]
        planar_distance = torch.linalg.norm(planar_delta, dim=-1)
        remaining_distance = float(planar_distance.max().item())

        _, _, current_yaw = math_utils.euler_xyz_from_quat(current_quat)
        desired_yaw = torch.where(
            planar_distance > 1.0e-6,
            torch.atan2(planar_delta[:, 1], planar_delta[:, 0]),
            current_yaw,
        )
        yaw_error = torch.atan2(torch.sin(desired_yaw - current_yaw), torch.cos(desired_yaw - current_yaw))

        if remaining_distance <= self.position_tolerance:
            snapped_pose = current_pose.clone()
            snapped_pose[:, 0] = self.goal_xyz[0]
            snapped_pose[:, 1] = self.goal_xyz[1]
            self.robot.write_root_pose_to_sim(snapped_pose)
            self.finished = True
            print("[Done] robot reached the goal.")
            return

        dir_x = float(planar_delta[0, 0].item()) / max(remaining_distance, 1.0e-6)
        dir_y = float(planar_delta[0, 1].item()) / max(remaining_distance, 1.0e-6)
        robot_xyz = self.get_xyz()

        if self.use_raycast_guard:
            hit = self._raycast_forward(
                robot_xyz[0],
                robot_xyz[1],
                dir_x,
                dir_y,
                RAYCAST_LOOKAHEAD,
            )
            if hit is not None:
                hit_distance, hit_path = hit
                self.blocked = True
                if hit_path != self._reported_hit_path:
                    print(f"[Raycast] blocking hit at {hit_distance:.3f} m on {hit_path}; stopping until clear.")
                    self._reported_hit_path = hit_path
                self.robot.write_root_pose_to_sim(current_pose)
                return

        if self.blocked:
            print("[Raycast] path cleared; resuming motion.")
        self.blocked = False
        self._reported_hit_path = None

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

        step_length = min(self.move_speed * dt, remaining_distance)
        step_scale = step_length / max(remaining_distance, 1.0e-6)
        next_pos = current_pos.clone()
        next_pos[:, :2] = current_pos[:, :2] + planar_delta * step_scale
        next_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(desired_yaw),
            torch.zeros_like(desired_yaw),
            desired_yaw,
        )
        next_pose = current_pose.clone()
        next_pose[:, :2] = next_pos[:, :2]
        next_pose[:, 3:7] = next_quat
        self.robot.write_root_pose_to_sim(next_pose)


def setup_camera() -> FloatingCamera:
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=[0, 0, 0.0],
        start_orientation=[0.62, 0.0, 0.0, -0.78],
        camera_height=1.5,
    )
    camera.init_manual()
    camera.reset()
    return camera


def spawn_obstacle_cylinder(stage) -> None:
    cylinder = UsdGeom.Cylinder.Define(stage, OBSTACLE_PRIM_PATH)
    cylinder.CreateRadiusAttr(OBSTACLE_RADIUS)
    cylinder.CreateHeightAttr(OBSTACLE_HEIGHT)
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.2, 0.15)])

    xform = UsdGeom.XformCommonAPI(cylinder.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*get_obstacle_center_xyz()))

    collision_api = (
        UsdPhysics.CollisionAPI(cylinder.GetPrim())
        if cylinder.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        else UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    )
    collision_api.CreateCollisionEnabledAttr(True)

    physx_collision_api = (
        PhysxSchema.PhysxCollisionAPI(cylinder.GetPrim())
        if cylinder.GetPrim().HasAPI(PhysxSchema.PhysxCollisionAPI)
        else PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
    )
    physx_collision_api.CreateContactOffsetAttr(0.02)
    physx_collision_api.CreateRestOffsetAttr(0.0)

    print(
        f"[Setup] spawned cylinder at {get_obstacle_center_xyz()} "
        f"(radius={OBSTACLE_RADIUS:.2f}, height={OBSTACLE_HEIGHT:.2f})."
    )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0, 0, 1.2], target=[0.0, 0.0, 0.0])

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    spawn_obstacle_cylinder(stage)

    camera = None if args.headless else setup_camera()

    scene = InteractiveScene(HomeSceneCfg(num_envs=1, env_spacing=2.5))
    robot = scene["g1"]

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())
    controller = RaycastGuardController(
        robot=robot,
        goal_xyz=ROBOT_GOAL,
        move_speed=args.move_speed,
        turn_speed_rad=math.radians(args.turn_speed_deg),
        position_tolerance=args.position_tolerance,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
        use_raycast_guard=RAYCAST_GUARD_ENABLED,
    )
    controller.initialize()
    scene.update(sim.get_physics_dt())

    print("[Setup] scene ready: G1 + blocking cylinder.")
    print(f"[Setup] goal={ROBOT_GOAL}")
    print(
        "[Setup] controller advances with pose writes; "
        "the forward ray guard freezes the current pose and resumes when the path clears."
    )
    print(f"[Setup] raycast_guard_enabled={RAYCAST_GUARD_ENABLED}")
    print(f"[Setup] raycast_heights={RAYCAST_HEIGHTS}")
    print(f"[Setup] raycast_lookahead={RAYCAST_LOOKAHEAD}")

    step_counter = 0
    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        controller.update(dt)
        sim.step(render=True)
        if camera is not None:
            camera.run(dt)
        scene.update(dt)
        step_counter += 1

        if step_counter % STATUS_PRINT_EVERY_STEPS == 0:
            print(
                f"[Status] step={step_counter}, robot={controller.get_xyz()}, "
                f"blocked={controller.blocked}"
            )

        if controller.finished:
            break

        if MAX_RUNTIME_STEPS > 0 and step_counter >= MAX_RUNTIME_STEPS:
            print(f"[Warn] stop after {MAX_RUNTIME_STEPS} steps.")
            break

    print(
        f"[Done] finished={controller.finished}, blocked={controller.blocked}, "
        f"final_robot_xyz={controller.get_xyz()}"
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
