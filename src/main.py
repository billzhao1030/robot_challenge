from __future__ import annotations

import math
import os
import shlex
from dataclasses import dataclass

from isaaclab.app import AppLauncher

from app_args import parse_main_args

# Human GoTo config for the combined script.
ENABLE_PEOPLE_GOTO = True
FULL_EXPERIENCE_PATH = "/home/xunyi/isaacsim5.1/apps/isaacsim.exp.full.kit"
PEOPLE_SCHEMA_STARTUP_EXTENSIONS = [
    "omni.usd.schema.anim",
    "omni.anim.graph.schema",
    "omni.usd.schema.omniscripting",
    "omni.anim.behavior.schema",
    "omni.anim.navigation.schema",
]
PEOPLE_RUNTIME_STARTUP_EXTENSIONS = [
    "omni.kit.scripting",
    "omni.anim.graph.bundle",
    "omni.anim.timeline",
    "omni.anim.navigation.bundle",
    "omni.anim.people",
]

# -------------------------
# 1) Parse args + launch app
# -------------------------
args = parse_main_args()

if ENABLE_PEOPLE_GOTO:
    if not getattr(args, "experience", ""):
        args.experience = FULL_EXPERIENCE_PATH

    existing_kit_args = shlex.split(getattr(args, "kit_args", "") or "")
    existing_ext_enables = {
        existing_kit_args[i + 1]
        for i in range(len(existing_kit_args) - 1)
        if existing_kit_args[i] == "--enable"
    }
    # Match the human GoTo startup behavior: load both schema and runtime
    # extensions up front so omni.anim.people is available immediately.
    for ext_name in PEOPLE_SCHEMA_STARTUP_EXTENSIONS + PEOPLE_RUNTIME_STARTUP_EXTENSIONS:
        if ext_name not in existing_ext_enables:
            existing_kit_args.extend(["--enable", ext_name])
    args.kit_args = " ".join(existing_kit_args)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -------------------------
# 2) Import Isaac Lab AFTER launching
# -------------------------
import carb
import isaaclab.sim as sim_utils
import omni.client
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaaclab.scene import InteractiveScene
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from camera.floating_camera import FloatingCamera
from controllers.waypoint_controller import WaypointController
from navigation.freemap_planner import FreemapPlanner, add_path_debug_vis, plan_waypoint_path
from world.grabber import ProximityGrabber, get_world_translation, vec3d_to_xyz
from world.scene_cfg import HomeSceneCfg

try:
    from isaacsim.core.utils.extensions import enable_extension
except Exception:
    from omni.isaac.core.utils.extensions import enable_extension


NAVIGATION_HEIGHT = 0.74
DEFAULT_WAYPOINT: tuple[float, float, float] = (-4.75, -3.08, NAVIGATION_HEIGHT)
TARGET_LOCATION: tuple[float, float, float] = (-0.65, 2.14, NAVIGATION_HEIGHT)

LEFT_HAND_BODY_NAME = "left_six_link"
ROBOT_PRIM_PATH = "/World/envs/env_0/Robot"
MONITORED_OBJECT_PRIM_PATH = "/World/envs/env_0/House/Meshes/studyroom_767841/ornament_0015"
PICK_DISTANCE_THRESHOLD = 0.8
PLACE_DISTANCE_THRESHOLD = 0.8
HAND_ATTACH_OFFSET = Gf.Vec3d(0.0, 0.0, -0.08)
GRABBED_OBJECT_USD_PATH = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/Meshes/ornament_0015.usd"

DEFAULT_ISAAC_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
CHARACTER_ROOT_PRIM_PATH = "/World/Characters"
CHARACTER_NAME = "Human"
CHARACTER_CONTROL_PRIM_PATH = f"{CHARACTER_ROOT_PRIM_PATH}/{CHARACTER_NAME}"
CHARACTER_MODEL_PRIM_PATH = f"{CHARACTER_CONTROL_PRIM_PATH}/Model"
HUMAN_SPAWN_XYZ: tuple[float, float, float] = (-1.70, -0.77, 0.0)
HUMAN_GOAL_XYZ: tuple[float, float, float] = (-3.55, 5.15, 0.0)
HUMAN_YAW_DEG = 0.0
HUMAN_GOAL_REACH_TOLERANCE = 0.2
HUMAN_MOVE_SPEED = 1.2
PEOPLE_STUCK_MAX_STEPS = 300
PEOPLE_STUCK_EPS = 1.0e-4

PEOPLE_EXTENSIONS = [
    "omni.anim.people",
    "omni.anim.navigation.bundle",
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
    "omni.anim.graph.ui",
    "omni.anim.graph.schema",
    "omni.kit.scripting",
]

# Task request config (set directly in code)
# AUTO_REQUEST_TASK: None | "navigate_to" | "pick_and_place"
# Keep None to leave the robot idle until you call task_manager.request_*() in code.
AUTO_REQUEST_TASK: str | None = None
NAVIGATE_TO_GOAL: tuple[float, float, float] = DEFAULT_WAYPOINT
PICK_OBJECT_LOCATION: tuple[float, float, float] | None = None
PICK_TARGET_LOCATION: tuple[float, float, float] = TARGET_LOCATION
PICK_STAGING_WAYPOINT: tuple[float, float, float] | None = DEFAULT_WAYPOINT

# Example sequence:
# after EXAMPLE_START_AFTER_SECONDS, run pick-and-place once, then navigate once.
EXAMPLE_ENABLE_DELAYED_PICK_THEN_NAV = True
EXAMPLE_START_AFTER_SECONDS = 1.0
EXAMPLE_NAVIGATE_GOAL: tuple[float, float, float] = (5.96, -1.54, NAVIGATION_HEIGHT)

# Optional execution gate for pending task requests.
WAIT_FOR_PRECONDITION = False
PRECONDITION_MIN_IDLE_STEPS = 200


def setup_camera():
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=[-5.46, -1.28, 0.0],
        start_orientation=[0.687852177796288, 0.0, 0.0, -0.7258507983745032],
        camera_height=1.3,
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


def get_robot_root_xyz(robot) -> tuple[float, float, float]:
    root_pos = robot.data.root_pos_w[0]
    return tuple(float(v) for v in root_pos.tolist())


def planar_distance_xy(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def plan_phase_waypoints(
    planner: FreemapPlanner,
    start_xyz: tuple[float, float, float],
    goal_xyz: tuple[float, float, float],
    approach_threshold: float | None = None,
) -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    approach_goal = (goal_xyz[0], goal_xyz[1], NAVIGATION_HEIGHT)
    if approach_threshold is not None:
        free_goal_xy = planner.find_free_point_within_radius(goal_xyz[0], goal_xyz[1], approach_threshold)
        if free_goal_xy is not None:
            approach_goal = (free_goal_xy[0], free_goal_xy[1], NAVIGATION_HEIGHT)
        else:
            fallback_goal = planner.find_nearest_reachable(goal_xyz[0], goal_xyz[1])
            if fallback_goal is None:
                raise RuntimeError(f"Could not find a reachable approach point near {goal_xyz}.")
            approach_goal = (fallback_goal[0], fallback_goal[1], NAVIGATION_HEIGHT)
            print(
                f"[Waypoint] no reachable point inside {approach_threshold:.2f} m of {goal_xyz}; "
                f"falling back to nearest reachable {approach_goal}."
            )

    return plan_waypoint_path(planner=planner, start_xyz=start_xyz, goals_xyz=[approach_goal]), approach_goal


class XformWaypointController:
    """Waypoint controller for the human control prim when People GoTo is unavailable."""

    def __init__(
        self,
        stage: Usd.Stage,
        prim_path: str,
        waypoints_xyz: list[tuple[float, float, float]],
        move_speed: float,
        turn_speed_deg: float,
        position_tolerance: float,
        yaw_tolerance_rad: float,
    ) -> None:
        self.stage = stage
        self.prim_path = prim_path
        self.move_speed = move_speed
        self.turn_speed_rad = math.radians(turn_speed_deg)
        self.position_tolerance = position_tolerance
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.waypoints: list[tuple[float, float, float]] = []
        self.current_index = 0
        self.finished = True

        self.prim = self.stage.GetPrimAtPath(self.prim_path)
        if not self.prim or not self.prim.IsValid():
            raise RuntimeError(f"Invalid prim path for fallback controller: {self.prim_path}")
        self.xform = UsdGeom.XformCommonAPI(self.prim)
        self.set_waypoints(waypoints_xyz)

    def set_waypoints(self, waypoints_xyz: list[tuple[float, float, float]]) -> None:
        self.waypoints = waypoints_xyz
        self.current_index = 0
        self.finished = len(waypoints_xyz) == 0
        print(f"[Human] loaded {len(waypoints_xyz)} waypoint(s).")

    def get_xyz(self) -> tuple[float, float, float]:
        x, y, z, _ = self._read_pose()
        return x, y, z

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _read_pose(self) -> tuple[float, float, float, float]:
        translate, rotate, _, _, _ = self.xform.GetXformVectors(Usd.TimeCode.Default())
        x = float(translate[0])
        y = float(translate[1])
        z = float(translate[2])
        yaw_rad = math.radians(float(rotate[2]))
        return x, y, z, yaw_rad

    def _write_pose(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        self.xform.SetTranslate(Gf.Vec3d(x, y, z))
        self.xform.SetRotate(
            Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )

    def update(self, dt: float) -> None:
        if self.finished:
            return

        x, y, z, yaw = self._read_pose()
        target = self.waypoints[self.current_index]
        dx = target[0] - x
        dy = target[1] - y
        planar_distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx) if planar_distance > 1.0e-6 else yaw
        yaw_error = self._wrap_to_pi(desired_yaw - yaw)

        if planar_distance <= self.position_tolerance:
            self._write_pose(target[0], target[1], target[2], desired_yaw)
            self.current_index += 1
            print(f"[Human] reached waypoint #{self.current_index}: {target}")
            if self.current_index >= len(self.waypoints):
                self.finished = True
                print("[Human] all waypoints completed.")
            return

        yaw_step = max(-self.turn_speed_rad * dt, min(self.turn_speed_rad * dt, yaw_error))
        if abs(yaw_error) > self.yaw_tolerance_rad:
            self._write_pose(x, y, z, yaw + yaw_step)
            return

        step = min(self.move_speed * dt, planar_distance)
        scale = step / max(planar_distance, 1.0e-6)
        self._write_pose(x + dx * scale, y + dy * scale, z, desired_yaw)


def enable_people_extensions() -> None:
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in PEOPLE_EXTENSIONS:
        if ext_manager.get_enabled_extension_id(ext_name) is not None:
            continue
        enable_extension(ext_name)
        simulation_app.update()


def ensure_isaac_asset_root() -> str:
    settings = carb.settings.get_settings()
    asset_root = settings.get("/persistent/isaac/asset_root/cloud")
    if asset_root:
        return str(asset_root).rstrip("/")
    settings.set("/persistent/isaac/asset_root/cloud", DEFAULT_ISAAC_ASSET_ROOT)
    print(f"[Config] set asset root to {DEFAULT_ISAAC_ASSET_ROOT}")
    return DEFAULT_ISAAC_ASSET_ROOT


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


def ensure_biped_setup(stage: Usd.Stage, people_assets_root: str) -> None:
    biped_setup_path = f"{CHARACTER_ROOT_PRIM_PATH}/Biped_Setup"
    biped_usd_path = f"{people_assets_root}/Biped_Setup.usd"

    root_prim = stage.GetPrimAtPath(CHARACTER_ROOT_PRIM_PATH)
    if not root_prim or not root_prim.IsValid():
        UsdGeom.Xform.Define(stage, CHARACTER_ROOT_PRIM_PATH)

    biped_prim = stage.GetPrimAtPath(biped_setup_path)
    if not biped_prim or not biped_prim.IsValid():
        biped_prim = stage.DefinePrim(biped_setup_path, "Xform")
        biped_prim.GetReferences().AddReference(biped_usd_path)

    UsdGeom.Imageable(biped_prim).MakeInvisible()


def spawn_human(stage: Usd.Stage, people_assets_root: str) -> str:
    human_usd_path = f"{people_assets_root}/F_Business_02/F_Business_02.usd"
    stat_result, _ = omni.client.stat(human_usd_path)
    if stat_result != omni.client.Result.OK:
        raise RuntimeError(f"Human USD not found: {human_usd_path}")

    control_prim = stage.DefinePrim(CHARACTER_CONTROL_PRIM_PATH, "Xform")
    model_prim = stage.DefinePrim(CHARACTER_MODEL_PRIM_PATH, "Xform")
    model_prim.GetReferences().AddReference(human_usd_path)

    xform = UsdGeom.XformCommonAPI(control_prim)
    xform.SetTranslate(Gf.Vec3d(*HUMAN_SPAWN_XYZ))
    xform.SetRotate(Gf.Vec3f(0.0, 0.0, HUMAN_YAW_DEG), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    print(f"[Human] spawned at {HUMAN_SPAWN_XYZ}")
    print(f"[Human] target goal: {HUMAN_GOAL_XYZ}")
    return CHARACTER_NAME


def find_animation_graph(stage: Usd.Stage):
    for prim in stage.Traverse():
        if prim.GetTypeName() == "AnimationGraph":
            return prim
    return None


def resolve_people_behavior_script_path() -> str:
    settings = carb.settings.get_settings()
    behavior_script_path = settings.get("/persistent/exts/omni.anim.people/behavior_script_settings/behavior_script_path")
    if behavior_script_path:
        return str(behavior_script_path)

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = ext_manager.get_enabled_extension_id("omni.anim.people")
    if ext_id is None:
        raise RuntimeError("omni.anim.people extension is not enabled.")
    ext_path = ext_manager.get_extension_path(ext_id)
    if not ext_path:
        raise RuntimeError("Failed to resolve omni.anim.people extension path.")

    behavior_script_path = os.path.join(ext_path, "omni", "anim", "people", "scripts", "character_behavior.py")
    settings.set(
        "/persistent/exts/omni.anim.people/behavior_script_settings/behavior_script_path",
        behavior_script_path,
    )
    return behavior_script_path


def setup_character_behavior(stage: Usd.Stage, behavior_script_path: str) -> list[str]:
    anim_graph_prim = find_animation_graph(stage)
    if anim_graph_prim is None:
        raise RuntimeError("AnimationGraph prim not found. Biped setup may be missing.")

    attached_paths: list[str] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "SkelRoot":
            continue
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(CHARACTER_CONTROL_PRIM_PATH):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        try:
            omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=[Sdf.Path(prim_path)])
        except Exception:
            pass

        omni.kit.commands.execute(
            "ApplyAnimationGraphAPICommand",
            paths=[Sdf.Path(prim_path)],
            animation_graph_path=Sdf.Path(str(anim_graph_prim.GetPath())),
        )
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim_path)])
        prim.GetAttribute("omni:scripting:scripts").Set([behavior_script_path])
        attached_paths.append(prim_path)

    if len(attached_paths) == 0:
        raise RuntimeError(f"No visible SkelRoot found under {CHARACTER_CONTROL_PRIM_PATH}")
    print(f"[Human] attached behavior script to {len(attached_paths)} SkelRoot prim(s).")
    return attached_paths


def write_command_file(command_lines: list[str]) -> tuple[str, str]:
    command_file_local = "/tmp/main_human_people_commands.txt"
    with open(command_file_local, "w", encoding="utf-8") as f:
        for line in command_lines:
            f.write(f"{line}\n")
    command_file_uri = f"file://{command_file_local}"
    print(f"[Human] wrote command file: {command_file_local}")
    return command_file_local, command_file_uri


def configure_people_settings(command_file_path: str) -> None:
    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.people/character_prim_path", CHARACTER_ROOT_PRIM_PATH)
    settings.set("/exts/omni.anim.people/command_settings/command_file_path", command_file_path)
    settings.set("/exts/omni.anim.people/command_settings/number_of_loop", 0)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", False)
    settings.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)


def get_people_character_xyz(character_prim_path: str) -> tuple[float, float, float]:
    import omni.anim.graph.core as ag

    character = ag.get_character(character_prim_path)
    if character is None:
        return (float("nan"), float("nan"), float("nan"))
    pos = carb.Float3(0.0, 0.0, 0.0)
    rot = carb.Float4(0.0, 0.0, 0.0, 1.0)
    character.get_world_transform(pos, rot)
    return float(pos.x), float(pos.y), float(pos.z)


class HumanGoToAgent:
    def __init__(
        self,
        *,
        stage: Usd.Stage,
        character_name: str,
        planner: FreemapPlanner | None,
        plan_debug_vis: bool,
        fallback_controller: XformWaypointController,
        position_tolerance: float,
    ) -> None:
        self.stage = stage
        self.character_name = character_name
        self.planner = planner
        self.plan_debug_vis = plan_debug_vis
        self.fallback_controller = fallback_controller
        self.position_tolerance = position_tolerance
        self.people_character_prim_path: str | None = None
        self.mode: str = "fallback"
        self.current_goal_xyz: tuple[float, float, float] | None = None
        self.startup_people_commands: list[str] = []
        self.prev_people_xyz: tuple[float, float, float] | None = None
        self.people_stuck_steps = 0

    def bind_people_backend(self, people_character_prim_path: str) -> None:
        self.people_character_prim_path = people_character_prim_path
        self.mode = "people"

    def bind_fallback_backend(self) -> None:
        self.mode = "fallback"
        self.people_character_prim_path = None

    def update(self, dt: float) -> None:
        if self.mode == "fallback":
            self.fallback_controller.update(dt)

    def get_current_xyz(self) -> tuple[float, float, float]:
        if self.mode == "people" and self.people_character_prim_path is not None:
            xyz = get_people_character_xyz(self.people_character_prim_path)
            if not any(v != v for v in xyz):
                return xyz
        return self.fallback_controller.get_xyz()

    def monitor_after_step(self) -> None:
        if self.mode != "people" or self.current_goal_xyz is None:
            return

        human_xyz = self.get_current_xyz()
        dist_to_goal = math.hypot(human_xyz[0] - self.current_goal_xyz[0], human_xyz[1] - self.current_goal_xyz[1])
        if dist_to_goal <= self.position_tolerance:
            self.prev_people_xyz = human_xyz
            self.people_stuck_steps = 0
            return

        if self.prev_people_xyz is not None:
            moved = math.hypot(human_xyz[0] - self.prev_people_xyz[0], human_xyz[1] - self.prev_people_xyz[1])
            if moved < PEOPLE_STUCK_EPS:
                self.people_stuck_steps += 1
            else:
                self.people_stuck_steps = 0
        self.prev_people_xyz = human_xyz

        if self.people_stuck_steps >= PEOPLE_STUCK_MAX_STEPS:
            print(
                f"[Warn] people GoTo appears stuck for {self.people_stuck_steps} steps at {human_xyz}; "
                "switching to fallback xform controller."
            )
            self.bind_fallback_backend()
            self.go_to(self.current_goal_xyz, force_interrupt=False)
            self.prev_people_xyz = None
            self.people_stuck_steps = 0

    def _plan_waypoints(self, goal_xyz: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        start_xyz = self.get_current_xyz()
        if self.planner is not None:
            waypoints = plan_waypoint_path(self.planner, start_xyz=start_xyz, goals_xyz=[goal_xyz])
            if len(waypoints) == 0:
                waypoints = [goal_xyz]
            if self.plan_debug_vis:
                add_path_debug_vis(self.stage, [start_xyz] + waypoints)
            print(f"[Human] occupancy waypoints planned: {len(waypoints)}")
            return waypoints
        return [goal_xyz]

    def _to_people_commands(self, waypoints_xyz: list[tuple[float, float, float]]) -> list[str]:
        return [f"{self.character_name} GoTo {x} {y} {z} _" for x, y, z in waypoints_xyz]

    def queue_startup_go_to(self, goal_xyz: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        waypoints = self._plan_waypoints(goal_xyz)
        self.startup_people_commands.extend(self._to_people_commands(waypoints))
        self.current_goal_xyz = goal_xyz
        return waypoints

    def flush_startup_command_file(self) -> tuple[str, str]:
        if len(self.startup_people_commands) == 0 and self.current_goal_xyz is not None:
            self.startup_people_commands = self._to_people_commands([self.current_goal_xyz])
        return write_command_file(self.startup_people_commands)

    def go_to(self, goal_xyz: tuple[float, float, float], force_interrupt: bool = True) -> list[tuple[float, float, float]]:
        waypoints = self._plan_waypoints(goal_xyz)
        self.current_goal_xyz = goal_xyz

        if self.mode == "people":
            try:
                from omni.anim.people.scripts.utils import Utils as PeopleUtils

                PeopleUtils.runtime_inject_command(
                    character_name=self.character_name,
                    command_list=self._to_people_commands(waypoints),
                    force_inject=force_interrupt,
                )
                print(f"[Human] runtime GoTo injected ({len(waypoints)} command line(s)).")
                return waypoints
            except Exception as exc:
                print(f"[Warn] runtime People GoTo injection failed; fallback to xform waypoints: {exc}")
                self.bind_fallback_backend()

        self.fallback_controller.set_waypoints(waypoints)
        return waypoints


@dataclass
class TaskRuntime:
    sim: sim_utils.SimulationContext
    scene: InteractiveScene
    camera: FloatingCamera | None
    robot: object
    planner: FreemapPlanner
    controller: WaypointController
    grabber: ProximityGrabber
    stage: object
    human_agent: HumanGoToAgent | None = None


@dataclass
class TaskRequest:
    name: str
    navigate_goal: tuple[float, float, float] | None = None
    object_location: tuple[float, float, float] | None = None
    target_location: tuple[float, float, float] | None = None
    staging_waypoint: tuple[float, float, float] | None = None


class RobotTaskManager:
    def __init__(self) -> None:
        self.pending_request: TaskRequest | None = None

    def request_navigate_to(self, goal_xyz: tuple[float, float, float]) -> None:
        self.pending_request = TaskRequest(name="navigate_to", navigate_goal=goal_xyz)
        print(f"[Task] navigate_to requested: {goal_xyz}")

    def request_pick_and_place(
        self,
        object_location: tuple[float, float, float],
        target_location: tuple[float, float, float],
        staging_waypoint: tuple[float, float, float] | None = None,
    ) -> None:
        self.pending_request = TaskRequest(
            name="pick_and_place",
            object_location=object_location,
            target_location=target_location,
            staging_waypoint=staging_waypoint,
        )
        print(f"[Task] pick_and_place requested: object={object_location}, target={target_location}")

    def has_pending(self) -> bool:
        return self.pending_request is not None

    def run_pending(self, runtime: TaskRuntime, plan_debug_vis: bool) -> tuple[str, bool] | None:
        request = self.pending_request
        if request is None:
            return None

        print(f"[Task] starting requested task: {request.name}")
        if request.name == "navigate_to":
            if request.navigate_goal is None:
                raise ValueError("navigate_to request missing goal.")
            success = navigate_to(
                runtime=runtime,
                plan_debug_vis=plan_debug_vis,
                goal_xyz=request.navigate_goal,
                label="navigate_to",
            )
        elif request.name == "pick_and_place":
            if request.object_location is None or request.target_location is None:
                raise ValueError("pick_and_place request missing object/target location.")
            success = pick_and_place(
                runtime=runtime,
                plan_debug_vis=plan_debug_vis,
                object_location=request.object_location,
                target_location=request.target_location,
                staging_waypoint=request.staging_waypoint,
            )
        else:
            raise ValueError(f"Unsupported task request: {request.name}")

        finished_name = request.name
        self.pending_request = None
        print(f"[Task] finished requested task: {finished_name}, success={success}")
        return finished_name, success


def precondition_met(idle_steps: int) -> bool:
    if not WAIT_FOR_PRECONDITION:
        return True
    return idle_steps >= PRECONDITION_MIN_IDLE_STEPS


def step_runtime(runtime: TaskRuntime) -> None:
    dt = runtime.sim.get_physics_dt()
    runtime.controller.update(dt)
    runtime.grabber.update()
    if runtime.human_agent is not None:
        runtime.human_agent.update(dt)
    runtime.sim.step(render=True)
    if runtime.camera is not None:
        runtime.camera.run(dt)
    runtime.scene.update(dt)
    if runtime.human_agent is not None:
        runtime.human_agent.monitor_after_step()


def load_route(
    runtime: TaskRuntime,
    plan_debug_vis: bool,
    start_pose_xyz: tuple[float, float, float],
    goal_xyz: tuple[float, float, float],
    label: str,
    approach_threshold: float | None = None,
) -> tuple[float, float, float]:
    route_waypoints, approach_goal = plan_phase_waypoints(
        planner=runtime.planner,
        start_xyz=start_pose_xyz,
        goal_xyz=goal_xyz,
        approach_threshold=approach_threshold,
    )
    runtime.controller.set_waypoints(route_waypoints)
    if plan_debug_vis:
        add_path_debug_vis(runtime.stage, [start_pose_xyz] + route_waypoints)
    print(f"[Waypoint] {label} goal: {goal_xyz}")
    print(f"[Waypoint] {label} approach goal: {approach_goal}")
    print(f"[Waypoint] {label} planned path: {runtime.controller.waypoints.tolist()}")
    return approach_goal


def navigate_to(
    runtime: TaskRuntime,
    plan_debug_vis: bool,
    goal_xyz: tuple[float, float, float],
    label: str = "navigate_to",
    approach_threshold: float | None = None,
    completion_threshold: float | None = None,
) -> bool:
    start_xyz = get_robot_root_xyz(runtime.robot)
    load_route(
        runtime=runtime,
        plan_debug_vis=plan_debug_vis,
        start_pose_xyz=start_xyz,
        goal_xyz=goal_xyz,
        label=label,
        approach_threshold=approach_threshold,
    )

    while simulation_app.is_running():
        robot_xyz = get_robot_root_xyz(runtime.robot)
        remaining_distance = planar_distance_xy(robot_xyz, goal_xyz)
        if completion_threshold is not None and remaining_distance <= completion_threshold:
            runtime.controller.set_waypoints([])
            print(f"[Task] {label}: reached threshold ({remaining_distance:.3f} m).")
            return True
        if runtime.controller.finished:
            if completion_threshold is None:
                print(f"[Task] {label}: navigation completed.")
                return True
            if remaining_distance <= completion_threshold:
                print(f"[Task] {label}: navigation completed inside threshold ({remaining_distance:.3f} m).")
                return True
            print(
                f"[Task] {label}: route ended at {robot_xyz}, which is still "
                f"{remaining_distance:.3f} m from the goal."
            )
            return False
        step_runtime(runtime)

    return False


def pick_and_place(
    runtime: TaskRuntime,
    plan_debug_vis: bool,
    object_location: tuple[float, float, float],
    target_location: tuple[float, float, float],
    staging_waypoint: tuple[float, float, float] | None = None,
) -> bool:
    if staging_waypoint is not None:
        navigate_to(
            runtime=runtime,
            plan_debug_vis=plan_debug_vis,
            goal_xyz=staging_waypoint,
            label="staging",
        )

    reached_object = navigate_to(
        runtime=runtime,
        plan_debug_vis=plan_debug_vis,
        goal_xyz=object_location,
        label="object",
        approach_threshold=PICK_DISTANCE_THRESHOLD,
        completion_threshold=PICK_DISTANCE_THRESHOLD,
    )
    if not reached_object:
        return False

    print("[Task] near object. Waiting for grab attach.")
    for _ in range(240):
        if not simulation_app.is_running():
            return False
        if runtime.grabber.is_attached:
            break
        step_runtime(runtime)
    if not runtime.grabber.is_attached:
        print("[Task] object was not attached within timeout.")
        return False

    reached_target = navigate_to(
        runtime=runtime,
        plan_debug_vis=plan_debug_vis,
        goal_xyz=target_location,
        label="target",
        approach_threshold=PLACE_DISTANCE_THRESHOLD,
        completion_threshold=PLACE_DISTANCE_THRESHOLD,
    )
    if not reached_target:
        return False

    target_distance = planar_distance_xy(get_robot_root_xyz(runtime.robot), target_location)
    if target_distance > PLACE_DISTANCE_THRESHOLD:
        print(
            f"[Task] not releasing object because distance to target is {target_distance:.3f} m "
            f"(threshold {PLACE_DISTANCE_THRESHOLD:.3f} m)."
        )
        return False

    runtime.controller.set_waypoints([])
    runtime.grabber.release_to_world(target_location)
    print(f"[Task] pick-and-place complete. Placed at {target_location}.")
    return True


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[0, 0, 1.2],
        target=[0.0, 0.0, 0.0],
    )

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")

    usd_path = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
    load_house_usd(env_index=0, usd_path=usd_path, usd_root_prim="/Root")
    add_room_lights(stage)

    camera = None if args.headless else setup_camera()

    scene = InteractiveScene(HomeSceneCfg(num_envs=1, env_spacing=2.5))
    robot = scene["g1"]

    # Initialize Isaac Lab robot handles before creating the waypoint controller.
    # Do this before spawning/configuring People to avoid the reset/GoTo conflict.
    sim.step(render=True)
    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())
    print("[Init] robot scene reset complete.")

    planner = FreemapPlanner(args.freemap_path, safety_margin_m=args.path_clearance)
    object_location = (
        PICK_OBJECT_LOCATION
        if PICK_OBJECT_LOCATION is not None
        else vec3d_to_xyz(get_world_translation(stage.GetPrimAtPath(MONITORED_OBJECT_PRIM_PATH)))
    )
    target_location = PICK_TARGET_LOCATION
    human_agent: HumanGoToAgent | None = None

    people_assets_root = f"{ensure_isaac_asset_root()}/Isaac/People/Characters"
    ensure_biped_setup(stage, people_assets_root)
    character_name = spawn_human(stage, people_assets_root)
    human_fallback_controller = XformWaypointController(
        stage=stage,
        prim_path=CHARACTER_CONTROL_PRIM_PATH,
        waypoints_xyz=[],
        move_speed=HUMAN_MOVE_SPEED,
        turn_speed_deg=180.0,
        position_tolerance=HUMAN_GOAL_REACH_TOLERANCE,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
    )
    human_agent = HumanGoToAgent(
        stage=stage,
        character_name=character_name,
        planner=planner,
        plan_debug_vis=args.plan_debug_vis,
        fallback_controller=human_fallback_controller,
        position_tolerance=HUMAN_GOAL_REACH_TOLERANCE,
    )

    if ENABLE_PEOPLE_GOTO:
        try:
            print("[Human Init] enabling People runtime extensions...")
            enable_people_extensions()
            print("[Human Init] planning startup GoTo...")
            planned_waypoints = human_agent.queue_startup_go_to(HUMAN_GOAL_XYZ)
            print(f"[Human] startup goal queued with {len(planned_waypoints)} waypoint(s).")
            print("[Human Init] writing command file...")
            _, command_file_uri = human_agent.flush_startup_command_file()
            read_result, _, _ = omni.client.read_file(command_file_uri)
            print(f"[Human] command file uri={command_file_uri}, read_result={read_result}")
            print("[Human Init] configuring People settings...")
            configure_people_settings(command_file_uri)
            print("[Human Init] resolving behavior script path...")
            behavior_script_path = resolve_people_behavior_script_path()
            print(f"[Human Init] behavior script path: {behavior_script_path}")
            print("[Human Init] attaching animation graph + behavior...")
            attached_paths = setup_character_behavior(stage, behavior_script_path)
            print(f"[Human Init] attached paths: {attached_paths}")
            human_agent.bind_people_backend(attached_paths[0])
            print("[Human Init] starting timeline...")
            omni.timeline.get_timeline_interface().play()
            print(f"[OK] using omni.anim.people GoTo behavior on {attached_paths[0]}.")
        except Exception as exc:
            print(f"[Warn] omni.anim.people setup failed, using fallback controller: {exc}")
            human_agent.bind_fallback_backend()
            human_agent.go_to(HUMAN_GOAL_XYZ)
            print("[OK] using fallback Xform waypoint controller.")
    else:
        human_agent.bind_fallback_backend()
        human_agent.go_to(HUMAN_GOAL_XYZ)
        print("[OK] using fallback Xform waypoint controller.")

    controller = WaypointController(
        robot=robot,
        waypoints_xyz=[],
        move_speed=args.move_speed,
        turn_speed_rad=math.radians(args.turn_speed_deg),
        position_tolerance=args.position_tolerance,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
    )
    controller.initialize()
    grabber = ProximityGrabber(
        stage=stage,
        robot=robot,
        hand_body_name=LEFT_HAND_BODY_NAME,
        object_prim_path=MONITORED_OBJECT_PRIM_PATH,
        trigger_distance=PICK_DISTANCE_THRESHOLD,
        grabbed_object_usd_path=GRABBED_OBJECT_USD_PATH,
        robot_prim_path=ROBOT_PRIM_PATH,
        hand_attach_offset=HAND_ATTACH_OFFSET,
    )

    runtime = TaskRuntime(
        sim=sim,
        scene=scene,
        camera=camera,
        robot=robot,
        planner=planner,
        controller=controller,
        grabber=grabber,
        stage=stage,
        human_agent=human_agent,
    )

    print("[OK] Scene ready: house + G1.")
    print(f"[Waypoint] default waypoint: {DEFAULT_WAYPOINT}")
    print(f"[Task] object location: {object_location}")
    print(f"[Task] target location: {target_location}")
    print(f"[Waypoint] freemap resolution: {planner.grid_resolution:.4f} m, safety margin: {args.path_clearance:.3f} m")
    print(f"[Grab] monitoring {MONITORED_OBJECT_PRIM_PATH}")
    if human_agent is not None:
        print(f"[Human] mode={human_agent.mode}, goal={HUMAN_GOAL_XYZ}")

    task_manager = RobotTaskManager()
    if AUTO_REQUEST_TASK == "navigate_to":
        task_manager.request_navigate_to(NAVIGATE_TO_GOAL)
    elif AUTO_REQUEST_TASK == "pick_and_place":
        task_manager.request_pick_and_place(
            object_location=object_location,
            target_location=target_location,
            staging_waypoint=PICK_STAGING_WAYPOINT,
        )
    elif AUTO_REQUEST_TASK is not None:
        raise ValueError(f"Unsupported AUTO_REQUEST_TASK: {AUTO_REQUEST_TASK}")

    if AUTO_REQUEST_TASK is None:
        print("[Task] no auto-requested task. Robot will stay idle until request_* is called.")
    if WAIT_FOR_PRECONDITION and task_manager.has_pending():
        print(
            f"[Task] pending request will wait for precondition: "
            f"idle_steps >= {PRECONDITION_MIN_IDLE_STEPS}"
        )
    if EXAMPLE_ENABLE_DELAYED_PICK_THEN_NAV:
        print(
            f"[Task] delayed example enabled: at t>={EXAMPLE_START_AFTER_SECONDS:.1f}s run pick_and_place, "
            f"then navigate_to {EXAMPLE_NAVIGATE_GOAL}"
        )

    idle_steps = 0
    wait_log_counter = 0
    elapsed_time_s = 0.0
    delayed_pick_requested = False
    delayed_nav_requested = False
    while simulation_app.is_running():
        if (
            EXAMPLE_ENABLE_DELAYED_PICK_THEN_NAV
            and not delayed_pick_requested
            and elapsed_time_s >= EXAMPLE_START_AFTER_SECONDS
            and not task_manager.has_pending()
        ):
            task_manager.request_pick_and_place(
                object_location=object_location,
                target_location=target_location,
                staging_waypoint=PICK_STAGING_WAYPOINT,
            )
            delayed_pick_requested = True

        if task_manager.has_pending():
            if precondition_met(idle_steps):
                result = task_manager.run_pending(runtime=runtime, plan_debug_vis=args.plan_debug_vis)
                if result is not None:
                    finished_name, success = result
                    if (
                        EXAMPLE_ENABLE_DELAYED_PICK_THEN_NAV
                        and finished_name == "pick_and_place"
                        and success
                        and not delayed_nav_requested
                    ):
                        task_manager.request_navigate_to(EXAMPLE_NAVIGATE_GOAL)
                        delayed_nav_requested = True
                wait_log_counter = 0
            else:
                wait_log_counter += 1
                if wait_log_counter % 240 == 0:
                    print(
                        f"[Task] waiting precondition for pending task: "
                        f"{idle_steps}/{PRECONDITION_MIN_IDLE_STEPS} idle steps"
                    )

        step_runtime(runtime)
        elapsed_time_s += runtime.sim.get_physics_dt()
        idle_steps += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
