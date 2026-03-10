from __future__ import annotations

import os
import shlex

from isaaclab.app import AppLauncher

from app_args import parse_main_args

# Enable this flag (or set HUMAN_ENABLE_PEOPLE_GOTO=1) to try omni.anim.people GoTo.
ENABLE_PEOPLE_GOTO = True
if os.environ.get("HUMAN_ENABLE_PEOPLE_GOTO") is not None:
    ENABLE_PEOPLE_GOTO = os.environ.get("HUMAN_ENABLE_PEOPLE_GOTO", "0") == "1"

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
    """Fallback controller when omni.anim.people behavior APIs are unavailable."""

    def __init__(
        self,
        stage: Usd.Stage,
        prim_path: str,
        goal_xyz: tuple[float, float, float],
        move_speed: float,
        turn_speed_deg: float,
        position_tolerance: float,
    ) -> None:
        self.stage = stage
        self.prim_path = prim_path
        self.goal_xyz = goal_xyz
        self.move_speed = move_speed
        self.turn_speed_rad = turn_speed_deg * 3.141592653589793 / 180.0
        self.position_tolerance = position_tolerance
        self.finished = False

        self.prim = self.stage.GetPrimAtPath(self.prim_path)
        if not self.prim or not self.prim.IsValid():
            raise RuntimeError(f"Invalid prim path for fallback controller: {self.prim_path}")
        self.xform = UsdGeom.XformCommonAPI(self.prim)

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
        dx = self.goal_xyz[0] - x
        dy = self.goal_xyz[1] - y
        planar_distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx) if planar_distance > 1.0e-6 else yaw
        yaw_error = self._wrap_to_pi(desired_yaw - yaw)

        if planar_distance <= self.position_tolerance:
            self._write_pose(self.goal_xyz[0], self.goal_xyz[1], self.goal_xyz[2], desired_yaw)
            self.finished = True
            return

        yaw_step = max(-self.turn_speed_rad * dt, min(self.turn_speed_rad * dt, yaw_error))
        if abs(yaw_error) > 0.1:
            self._write_pose(x, y, z, yaw + yaw_step)
            return

        step = min(self.move_speed * dt, planar_distance)
        scale = step / max(planar_distance, 1.0e-6)
        self._write_pose(x + dx * scale, y + dy * scale, z, desired_yaw)


def setup_camera() -> FloatingCamera:
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=[0, 0, 0.0],
        start_orientation=[0.687852177796288, 0.0, 0.0, -0.7258507983745032],
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


def write_command_file(character_name: str, goal_xyz: tuple[float, float, float]) -> tuple[str, str]:
    command_file_local = "/tmp/human_people_commands.txt"
    with open(command_file_local, "w", encoding="utf-8") as f:
        f.write(f"{character_name} GoTo {goal_xyz[0]} {goal_xyz[1]} {goal_xyz[2]} _\n")
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

    fallback_controller = XformWaypointController(
        stage=stage,
        prim_path=CHARACTER_CONTROL_PRIM_PATH,
        goal_xyz=HUMAN_GOAL_XYZ,
        move_speed=1.2,
        turn_speed_deg=180.0,
        position_tolerance=GOAL_REACH_TOLERANCE,
    )
    using_people_behavior = False
    people_character_prim_path: str | None = None
    prev_people_xyz: tuple[float, float, float] | None = None
    people_stuck_steps = 0

    if ENABLE_PEOPLE_GOTO:
        try:
            enable_people_extensions()
            command_file_local, command_file_uri = write_command_file(character_name, HUMAN_GOAL_XYZ)
            read_result, _, _ = omni.client.read_file(command_file_uri)
            print(f"[Human] command file uri={command_file_uri}, read_result={read_result}")
            configure_people_settings(command_file_uri)
            behavior_script_path = resolve_people_behavior_script_path()
            attached_paths = setup_character_behavior(stage, behavior_script_path)
            people_character_prim_path = attached_paths[0]
            using_people_behavior = True
            fallback_controller = None
            print(f"[OK] using omni.anim.people GoTo behavior on {people_character_prim_path}.")
        except Exception as exc:
            print(f"[Warn] omni.anim.people setup failed, using fallback controller: {exc}")
            print("[OK] using fallback Xform waypoint controller.")
    else:
        print("[Info] ENABLE_PEOPLE_GOTO=False, skipping omni.anim.people setup.")
        print("[OK] using fallback Xform waypoint controller.")

    sim.reset()
    omni.timeline.get_timeline_interface().play()
    print("[OK] human scene started.")

    step_count = 0
    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        if fallback_controller is not None:
            fallback_controller.update(dt)
        sim.step(render=True)
        if camera is not None:
            camera.run(dt)
        step_count += 1

        if using_people_behavior and people_character_prim_path is not None:
            human_xyz = get_people_character_xyz(people_character_prim_path)
        else:
            human_xyz = get_control_world_xyz(stage)
        dist_to_goal = ((human_xyz[0] - HUMAN_GOAL_XYZ[0]) ** 2 + (human_xyz[1] - HUMAN_GOAL_XYZ[1]) ** 2) ** 0.5

        # Some Isaac Sim 5.1 setups initialize omni.anim.people successfully but never advance the character.
        # Detect this condition and degrade to deterministic xform navigation so tests still complete.
        if using_people_behavior and people_character_prim_path is not None and dist_to_goal > GOAL_REACH_TOLERANCE:
            if prev_people_xyz is not None:
                moved = ((human_xyz[0] - prev_people_xyz[0]) ** 2 + (human_xyz[1] - prev_people_xyz[1]) ** 2) ** 0.5
                if moved < PEOPLE_STUCK_EPS:
                    people_stuck_steps += 1
                else:
                    people_stuck_steps = 0
            prev_people_xyz = human_xyz

            if people_stuck_steps >= PEOPLE_STUCK_MAX_STEPS and fallback_controller is None:
                print(
                    f"[Warn] people GoTo appears stuck for {people_stuck_steps} steps at {human_xyz}; "
                    "switching to fallback xform controller."
                )
                fallback_controller = XformWaypointController(
                    stage=stage,
                    prim_path=CHARACTER_CONTROL_PRIM_PATH,
                    goal_xyz=HUMAN_GOAL_XYZ,
                    move_speed=1.2,
                    turn_speed_deg=180.0,
                    position_tolerance=GOAL_REACH_TOLERANCE,
                )
                using_people_behavior = False
                people_character_prim_path = None
                prev_people_xyz = None

        if step_count % STATUS_PRINT_EVERY_STEPS == 0:
            mode = "people" if using_people_behavior else "fallback"
            print(f"[Status] step={step_count}, mode={mode}, human_xyz={human_xyz}, dist_to_goal={dist_to_goal:.3f}")

        if dist_to_goal <= GOAL_REACH_TOLERANCE:
            print(f"[Done] goal reached with dist={dist_to_goal:.3f}")
            break

        if step_count >= MAX_RUNTIME_STEPS:
            print(f"[Warn] stop at {MAX_RUNTIME_STEPS} steps before reaching goal.")

    simulation_app.close()


if __name__ == "__main__":
    main()
