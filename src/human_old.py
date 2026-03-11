from __future__ import annotations

import math
import os
import shlex

from isaaclab.app import AppLauncher

from app_args import parse_main_args

# Enable this flag (or set HUMAN_ENABLE_PEOPLE_GOTO=1) to try omni.anim.people GoTo.
ENABLE_PEOPLE_GOTO = True
PERFORM_SIM_RESET = os.environ.get("HUMAN_PERFORM_SIM_RESET", "0") == "1"

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
    # These schema extensions need to be present at Kit startup, not enabled late.
    if not getattr(args, "experience", ""):
        args.experience = FULL_EXPERIENCE_PATH

    existing_kit_args = shlex.split(getattr(args, "kit_args", "") or "")
    existing_ext_enables = {
        existing_kit_args[i + 1]
        for i in range(len(existing_kit_args) - 1)
        if existing_kit_args[i] == "--enable"
    }
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
from pxr import Gf, Sdf, Usd, UsdGeom

from camera.floating_camera import FloatingCamera
from navigation.freemap_planner import FreemapPlanner, add_path_debug_vis, plan_waypoint_path

try:
    from isaacsim.core.utils.extensions import enable_extension
except Exception:
    from omni.isaac.core.utils.extensions import enable_extension


DEFAULT_ISAAC_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
CHARACTER_ROOT_PRIM_PATH = "/World/Characters"
CHARACTER_NAME = "Human"
CHARACTER_CONTROL_PRIM_PATH = f"{CHARACTER_ROOT_PRIM_PATH}/{CHARACTER_NAME}"
CHARACTER_MODEL_PRIM_PATH = f"{CHARACTER_CONTROL_PRIM_PATH}/Model"
HOUSE_USD_PATH = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
HOUSE_USD_ROOT_PRIM = "/Root"
HOUSE_PRIM_PATH = "/World/envs/env_0/House"

HUMAN_SPAWN_XYZ: tuple[float, float, float] = (0.0, 0.0, 0.0)
HUMAN_GOAL_XYZ: tuple[float, float, float] = (-4.75, -3.08, 0.0)
HUMAN_YAW_DEG = 0.0
GOAL_REACH_TOLERANCE = 0.2
USE_OCCUPANCY_WAYPOINTS = True
# Example runtime command trigger:
# RUNTIME_TEXT_COMMANDS = [(600, "go -2.8 -1.0 0")]
RUNTIME_TEXT_COMMANDS: list[tuple[int, str]] = []

STATUS_PRINT_EVERY_STEPS = 120
MAX_RUNTIME_STEPS = 20000
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


class XformWaypointController:
    """Waypoint controller for a plain Xform prim."""

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
        self.turn_speed_rad = turn_speed_deg * 3.141592653589793 / 180.0
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
        import math

        return math.atan2(math.sin(angle), math.cos(angle))

    def _read_pose(self) -> tuple[float, float, float, float]:
        import math

        translate, rotate, _, _, _ = self.xform.GetXformVectors(Usd.TimeCode.Default())
        x = float(translate[0])
        y = float(translate[1])
        z = float(translate[2])
        yaw_rad = math.radians(float(rotate[2]))
        return x, y, z, yaw_rad

    def _write_pose(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        import math

        self.xform.SetTranslate(Gf.Vec3d(x, y, z))
        self.xform.SetRotate(
            Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )

    def update(self, dt: float) -> None:
        import math

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


def setup_camera() -> FloatingCamera:
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=[0, 0, 0],
        start_orientation=[0, 0.0, 0.0, 0.0],
        camera_height=1.3,
    )
    camera.init_manual()
    camera.reset()
    return camera


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


def setup_basic_scene() -> None:
    try:
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    except Exception as exc:
        # Fallback when ISAAC_NUCLEUS_DIR resolves to None in this process.
        carb.log_warn(f"GroundPlaneCfg unavailable, using local fallback floor: {exc}")
        floor_cfg = sim_utils.CuboidCfg(
            size=(50.0, 50.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        )
        floor_cfg.func("/World/FallbackGround", floor_cfg, translation=(0.0, 0.0, -0.05))

    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0)
    cfg_light.func("/World/DomeLight", cfg_light)


def load_house_usd(stage: Usd.Stage, usd_path: str, usd_root_prim: str) -> None:
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    house_prim = stage.DefinePrim(HOUSE_PRIM_PATH, "Xform")
    house_prim.GetReferences().AddReference(usd_path, usd_root_prim)
    print(f"[OK] House referenced at {HOUSE_PRIM_PATH} from {usd_path} ({usd_root_prim})")


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

    attached = 0
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

        script_attr = prim.GetAttribute("omni:scripting:scripts")
        script_attr.Set([behavior_script_path])
        attached += 1
        attached_paths.append(prim_path)

    if attached == 0:
        raise RuntimeError(f"No visible SkelRoot found under {CHARACTER_CONTROL_PRIM_PATH}")
    print(f"[Human] attached behavior script to {attached} SkelRoot prim(s).")
    return attached_paths


def write_command_file(command_lines: list[str]) -> tuple[str, str]:
    command_file_local = "/tmp/human_people_commands.txt"
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

def get_control_world_xyz(stage: Usd.Stage) -> tuple[float, float, float]:
    prim = stage.GetPrimAtPath(CHARACTER_CONTROL_PRIM_PATH)
    if not prim or not prim.IsValid():
        return (float("nan"), float("nan"), float("nan"))
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


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
    """
    Modular human navigation API.

    Main entry points:
    - go_to((x, y, z)): runtime command.
    - go_to_from_text("go x y z"): runtime text command parser.
    """

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

    def _plan_waypoints(self, goal_xyz: tuple[float, float, float], use_occupancy: bool) -> list[tuple[float, float, float]]:
        start_xyz = self.get_current_xyz()
        if self.planner is not None and use_occupancy:
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

    def queue_startup_go_to(self, goal_xyz: tuple[float, float, float], use_occupancy: bool = True) -> list[tuple[float, float, float]]:
        waypoints = self._plan_waypoints(goal_xyz, use_occupancy=use_occupancy)
        self.startup_people_commands.extend(self._to_people_commands(waypoints))
        self.current_goal_xyz = goal_xyz
        return waypoints

    def flush_startup_command_file(self) -> tuple[str, str]:
        if len(self.startup_people_commands) == 0 and self.current_goal_xyz is not None:
            self.startup_people_commands = self._to_people_commands([self.current_goal_xyz])
        return write_command_file(self.startup_people_commands)

    def go_to(
        self,
        goal_xyz: tuple[float, float, float],
        *,
        use_occupancy: bool = True,
        force_interrupt: bool = True,
    ) -> list[tuple[float, float, float]]:
        waypoints = self._plan_waypoints(goal_xyz, use_occupancy=use_occupancy)
        self.current_goal_xyz = goal_xyz

        if self.mode == "people":
            try:
                from omni.anim.people.scripts.utils import Utils as PeopleUtils

                cmd_lines = self._to_people_commands(waypoints)
                PeopleUtils.runtime_inject_command(
                    character_name=self.character_name,
                    command_list=cmd_lines,
                    force_inject=force_interrupt,
                )
                print(f"[Human] runtime GoTo injected ({len(cmd_lines)} command line(s)).")
                return waypoints
            except Exception as exc:
                print(f"[Warn] runtime People GoTo injection failed; fallback to xform waypoints: {exc}")
                self.bind_fallback_backend()

        self.fallback_controller.set_waypoints(waypoints)
        return waypoints

    def go_to_from_text(self, text: str, *, use_occupancy: bool = True) -> list[tuple[float, float, float]]:
        normalized = text.replace(",", " ").strip().lower()
        tokens = [w for w in normalized.split() if w]
        if len(tokens) < 3 or tokens[0] not in {"go", "goto", "go_to"}:
            raise ValueError(f"Unsupported command: {text}. Expected: 'go x y z'")
        x = float(tokens[1])
        y = float(tokens[2])
        z = float(tokens[3]) if len(tokens) >= 4 else self.get_current_xyz()[2]
        return self.go_to((x, y, z), use_occupancy=use_occupancy)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 1.0])

    asset_root = ensure_isaac_asset_root()
    people_assets_root = f"{asset_root}/Isaac/People/Characters"

    setup_basic_scene()
    stage = omni.usd.get_context().get_stage()
    camera = None if args.headless else setup_camera()
    load_house_usd(stage=stage, usd_path=HOUSE_USD_PATH, usd_root_prim=HOUSE_USD_ROOT_PRIM)

    ensure_biped_setup(stage, people_assets_root)
    character_name = spawn_human(stage, people_assets_root)

    planner: FreemapPlanner | None = None
    if USE_OCCUPANCY_WAYPOINTS:
        try:
            planner = FreemapPlanner(args.freemap_path, safety_margin_m=args.path_clearance)
            print(
                f"[Human] occupancy planner loaded: {args.freemap_path}, "
                f"resolution={planner.grid_resolution:.4f} m, clearance={args.path_clearance:.3f} m"
            )
        except Exception as exc:
            print(f"[Warn] failed to load occupancy planner ({args.freemap_path}): {exc}")

    fallback_controller = XformWaypointController(
        stage=stage,
        prim_path=CHARACTER_CONTROL_PRIM_PATH,
        waypoints_xyz=[],
        move_speed=1.2,
        turn_speed_deg=180.0,
        position_tolerance=GOAL_REACH_TOLERANCE,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
    )
    human_agent = HumanGoToAgent(
        stage=stage,
        character_name=character_name,
        planner=planner,
        plan_debug_vis=args.plan_debug_vis,
        fallback_controller=fallback_controller,
        position_tolerance=GOAL_REACH_TOLERANCE,
    )
    prev_people_xyz: tuple[float, float, float] | None = None
    people_stuck_steps = 0

    if ENABLE_PEOPLE_GOTO:
        try:
            enable_people_extensions()
            planned_waypoints = human_agent.queue_startup_go_to(HUMAN_GOAL_XYZ, use_occupancy=USE_OCCUPANCY_WAYPOINTS)
            print(f"[Human] startup goal queued with {len(planned_waypoints)} waypoint(s).")
            command_file_local, command_file_uri = human_agent.flush_startup_command_file()
            read_result, _, _ = omni.client.read_file(command_file_uri)
            print(f"[Human] command file uri={command_file_uri}, read_result={read_result}")
            configure_people_settings(command_file_uri)
            behavior_script_path = resolve_people_behavior_script_path()
            attached_paths = setup_character_behavior(stage, behavior_script_path)
            human_agent.bind_people_backend(attached_paths[0])
            print(f"[OK] using omni.anim.people GoTo behavior on {attached_paths[0]}.")
        except Exception as exc:
            print(f"[Warn] omni.anim.people setup failed, using fallback controller: {exc}")
            human_agent.bind_fallback_backend()
            human_agent.go_to(HUMAN_GOAL_XYZ, use_occupancy=USE_OCCUPANCY_WAYPOINTS)
            print("[OK] using fallback Xform waypoint controller.")
    else:
        print("[Info] ENABLE_PEOPLE_GOTO=False, skipping omni.anim.people setup.")
        human_agent.bind_fallback_backend()
        human_agent.go_to(HUMAN_GOAL_XYZ, use_occupancy=USE_OCCUPANCY_WAYPOINTS)
        print("[OK] using fallback Xform waypoint controller.")

    if PERFORM_SIM_RESET:
        print("[Init] resetting physics before run loop...")
        sim.reset()
        print("[Init] physics reset complete.")
    else:
        print("[Init] skipping sim.reset() (set HUMAN_PERFORM_SIM_RESET=1 to enable).")
    omni.timeline.get_timeline_interface().play()
    print("[OK] human scene started.")
    print("[Human] modular command API ready: human_agent.go_to((x, y, z)) / human_agent.go_to_from_text('go x y z').")

    step_count = 0
    fired_text_commands: set[int] = set()
    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        human_agent.update(dt)
        sim.step(render=True)
        if camera is not None:
            camera.run(dt)
        step_count += 1

        for trigger_step, command_text in RUNTIME_TEXT_COMMANDS:
            if trigger_step == step_count and trigger_step not in fired_text_commands:
                human_agent.go_to_from_text(command_text, use_occupancy=USE_OCCUPANCY_WAYPOINTS)
                fired_text_commands.add(trigger_step)

        human_xyz = human_agent.get_current_xyz()
        goal_xyz = human_agent.current_goal_xyz if human_agent.current_goal_xyz is not None else HUMAN_GOAL_XYZ
        dist_to_goal = ((human_xyz[0] - goal_xyz[0]) ** 2 + (human_xyz[1] - goal_xyz[1]) ** 2) ** 0.5

        # Some Isaac Sim 5.1 setups initialize omni.anim.people successfully but never advance the character.
        # Detect this condition and degrade to deterministic xform navigation so tests still complete.
        if human_agent.mode == "people" and dist_to_goal > GOAL_REACH_TOLERANCE:
            if prev_people_xyz is not None:
                moved = ((human_xyz[0] - prev_people_xyz[0]) ** 2 + (human_xyz[1] - prev_people_xyz[1]) ** 2) ** 0.5
                if moved < PEOPLE_STUCK_EPS:
                    people_stuck_steps += 1
                else:
                    people_stuck_steps = 0
            prev_people_xyz = human_xyz

            if people_stuck_steps >= PEOPLE_STUCK_MAX_STEPS:
                print(
                    f"[Warn] people GoTo appears stuck for {people_stuck_steps} steps at {human_xyz}; "
                    "switching to fallback xform controller."
                )
                human_agent.bind_fallback_backend()
                human_agent.go_to(goal_xyz, use_occupancy=USE_OCCUPANCY_WAYPOINTS, force_interrupt=False)
                prev_people_xyz = None
                people_stuck_steps = 0

        if step_count % STATUS_PRINT_EVERY_STEPS == 0:
            mode = human_agent.mode
            print(f"[Status] step={step_count}, mode={mode}, human_xyz={human_xyz}, dist_to_goal={dist_to_goal:.3f}")

        if dist_to_goal <= GOAL_REACH_TOLERANCE:
            print(f"[Done] goal reached with dist={dist_to_goal:.3f}")
            break

        if step_count >= MAX_RUNTIME_STEPS:
            print(f"[Warn] stop at {MAX_RUNTIME_STEPS} steps before reaching goal.")
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
