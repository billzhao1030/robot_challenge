from __future__ import annotations

import os
import math

from isaaclab.app import AppLauncher

from app_args import parse_main_args


args = parse_main_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import numpy as np
import omni.usd
import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG
from pxr import Gf, Usd, UsdGeom, UsdLux

from camera.floating_camera import FloatingCamera


HOUSE_USD_PATH = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
HOUSE_USD_ROOT_PRIM = "/Root"
LOCAL_G1_USD_PATH = "/home/xunyi/Downloads/g1.usd"


@configclass
class StaticG1SceneCfg(InteractiveSceneCfg):
    g1: ArticulationCfg = G1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.9, -0.5, 0.71),
            rot=(0.50, 0.0, 0.0, -0.866),
        ),
    )
    if os.path.exists(LOCAL_G1_USD_PATH):
        g1.spawn.usd_path = LOCAL_G1_USD_PATH
    g1.spawn.rigid_props.disable_gravity = True
    g1.spawn.articulation_props.fix_root_link = False


def load_house_usd(stage: Usd.Stage) -> None:
    house_prim_path = "/World/envs/env_0/House"
    house_prim = stage.DefinePrim(house_prim_path, "Xform")
    house_prim.GetReferences().AddReference(HOUSE_USD_PATH, HOUSE_USD_ROOT_PRIM)
    print(f"[OK] House referenced at {house_prim_path} from {HOUSE_USD_PATH} ({HOUSE_USD_ROOT_PRIM})")


def add_room_lights(stage: Usd.Stage) -> None:
    key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key_light.CreateIntensityAttr(2500.0)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    key_xform = UsdGeom.XformCommonAPI(key_light.GetPrim())
    key_xform.SetRotate(Gf.Vec3f(45.0, -30.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    fill_light = UsdLux.SphereLight.Define(stage, "/World/FillLight")
    fill_light.CreateIntensityAttr(15000.0)
    fill_light.CreateRadiusAttr(0.7)
    fill_light.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
    fill_xform = UsdGeom.XformCommonAPI(fill_light.GetPrim())
    fill_xform.SetTranslate(Gf.Vec3d(-0.5, 0.8, 2.6))


def setup_camera() -> FloatingCamera:
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=[-2.5, 2.26, 0.0],
        start_orientation=[-0.707, 0.0, 0.0, 0.707],
        camera_height=1.3,
    )
    camera.init_manual()
    camera.reset()
    return camera


def quat_wxyz_to_yaw(quat_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


class RobotKeyboardPoseController:
    def __init__(self, robot, camera: FloatingCamera | None) -> None:
        import carb

        self.robot = robot
        self.camera = camera
        self.robot_mode = False
        self.move_speed = 0.75
        self.turn_speed = math.radians(45.0)

        root_state = self.robot.data.default_root_state[0].detach().clone()
        self.position = np.array(
            [float(root_state[0]), float(root_state[1]), float(root_state[2])],
            dtype=np.float64,
        )
        self.yaw = quat_wxyz_to_yaw(
            (
                float(root_state[3]),
                float(root_state[4]),
                float(root_state[5]),
                float(root_state[6]),
            )
        )

        # [forward, strafe, yaw]
        self.command = np.zeros(3, dtype=np.float64)
        self._pressed_keys: set[str] = set()
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        try:
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
        except Exception:
            app_window = None
        if app_window is not None:
            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

    def _set_robot_mode(self, enabled: bool) -> None:
        y_is_down = "Y" in self._pressed_keys
        self._pressed_keys.clear()
        if y_is_down:
            self._pressed_keys.add("Y")
        self.robot_mode = bool(enabled)
        self.command[:] = 0.0
        if self.camera is not None:
            self.camera.set_input_enabled(not self.robot_mode)
        mode_name = "robot" if self.robot_mode else "camera"
        print(f"[Control] keyboard mode: {mode_name}")

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        import carb

        key_name = str(getattr(event.input, "name", event.input)).split(".")[-1].upper()
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and key_name == "Y":
            if key_name not in self._pressed_keys:
                self._pressed_keys.add(key_name)
                self._set_robot_mode(not self.robot_mode)
            return True
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE and key_name == "Y":
            self._pressed_keys.discard(key_name)
            return True

        if not self.robot_mode:
            return True

        key_to_command = {
            "W": (1.0, 0.0, 0.0),
            "S": (-1.0, 0.0, 0.0),
            "A": (0.0, -1.0, 0.0),
            "D": (0.0, 1.0, 0.0),
            "J": (0.0, 0.0, 1.0),
            "L": (0.0, 0.0, -1.0),
        }
        if key_name not in key_to_command:
            return True

        delta = np.array(key_to_command[key_name], dtype=np.float64)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name not in self._pressed_keys:
                self._pressed_keys.add(key_name)
                self.command += delta
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self._pressed_keys:
                self._pressed_keys.remove(key_name)
                self.command -= delta
        return True

    def update(self, dt: float) -> None:
        if self.robot_mode:
            forward_cmd, strafe_cmd, yaw_cmd = self.command
            self.yaw += yaw_cmd * self.turn_speed * dt

            forward_dir = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0], dtype=np.float64)
            right_dir = np.array([math.sin(self.yaw), -math.cos(self.yaw), 0.0], dtype=np.float64)
            self.position += (forward_dir * forward_cmd + right_dir * strafe_cmd) * self.move_speed * dt

        self.apply_pose()

    def apply_pose(self) -> None:
        root_pose = self.robot.data.default_root_state[:, :7].clone()
        root_pose[:, 0] = float(self.position[0])
        root_pose[:, 1] = float(self.position[1])
        root_pose[:, 2] = float(self.position[2])
        root_pose[:, 3:7] = torch.tensor(
            yaw_to_quat_wxyz(self.yaw),
            dtype=root_pose.dtype,
            device=root_pose.device,
        )
        self.robot.write_root_pose_to_sim(root_pose)


def freeze_g1_default_pose(robot, pose_controller: RobotKeyboardPoseController, *, reset: bool = False) -> None:
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    zero_root_velocity = robot.data.default_root_state[:, 7:13].clone()
    zero_root_velocity[:] = 0.0

    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    pose_controller.apply_pose()
    robot.write_root_velocity_to_sim(zero_root_velocity)
    robot.set_joint_position_target(default_joint_pos)
    robot.write_data_to_sim()
    if reset:
        robot.reset()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[-5.46, -1.28, 1.3], target=[-0.7, 0.0, 1.0])

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    load_house_usd(stage)
    add_room_lights(stage)

    camera = None if args.headless else setup_camera()
    scene = InteractiveScene(StaticG1SceneCfg(num_envs=1, env_spacing=2.5))
    robot = scene["g1"]

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())
    pose_controller = RobotKeyboardPoseController(robot=robot, camera=camera)
    freeze_g1_default_pose(robot, pose_controller, reset=True)
    print("[Setup] static scene ready: house USD + lights + G1 + floating camera.")
    print("[Control] press Y to switch between camera and robot keyboard control.")
    print("[Control] robot mode: W/S move forward/back, A/D strafe, J/L yaw rotate.")

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        pose_controller.update(dt)
        freeze_g1_default_pose(robot, pose_controller)

        sim.step(render=True)
        freeze_g1_default_pose(robot, pose_controller)
        if camera is not None:
            camera.run(dt)
        scene.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
