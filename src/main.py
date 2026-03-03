from __future__ import annotations

import argparse
import math
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
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -------------------------
# 2) Import Isaac Lab AFTER launching
# -------------------------
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG

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


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
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

    controller = WaypointController(
        robot=robot,
        waypoints_xyz=parse_waypoints(args.waypoints),
        move_speed=args.move_speed,
        turn_speed_rad=math.radians(args.turn_speed_deg),
        position_tolerance=args.position_tolerance,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
    )
    controller.initialize()

    print("[OK] Scene ready: house + G1.")
    print(f"[Waypoint] target sequence: {controller.waypoints.tolist()}")

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
