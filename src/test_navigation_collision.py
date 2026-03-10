from __future__ import annotations

import math

from isaaclab.app import AppLauncher

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
import carb
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from pxr import Gf, Sdf, Usd, UsdGeom

from camera.floating_camera import FloatingCamera
from controllers.waypoint_controller import WaypointController
from navigation.freemap_planner import FreemapPlanner, add_path_debug_vis, plan_waypoint_path


NAVIGATION_HEIGHT = 0.74

ROBOT_GOAL: tuple[float, float, float] = (-4.75, -3.08, NAVIGATION_HEIGHT)

HUMAN_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd"
)
HUMAN_PRIM_PATH = "/World/envs/env_0/Human"
HUMAN_MODEL_PRIM_PATH = f"{HUMAN_PRIM_PATH}/Model"
HUMAN_SPAWN_XYZ: tuple[float, float, float] = (-1.0, -0.8, 0.0)
HUMAN_GOAL_XYZ: tuple[float, float, float] = (-2.8, -1.0, 0.0)
HUMAN_MOVE_SPEED = 1.2
HUMAN_TURN_SPEED_RAD = math.radians(180.0)

HOUSE_USD_PATH = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
HOUSE_USD_ROOT_PRIM = "/Root"
DEFAULT_ISAAC_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

STATUS_PRINT_EVERY_STEPS = 120
MAX_RUNTIME_STEPS = 20000


class XformWaypointController:
    """Simple waypoint controller for a plain Xform prim (no IRA, no animation graph)."""

    def __init__(
        self,
        stage: Usd.Stage,
        prim_path: str,
        waypoints_xyz: list[tuple[float, float, float]],
        move_speed: float,
        turn_speed_rad: float,
        position_tolerance: float,
        yaw_tolerance_rad: float,
    ) -> None:
        self.stage = stage
        self.prim_path = prim_path
        self.move_speed = move_speed
        self.turn_speed_rad = turn_speed_rad
        self.position_tolerance = position_tolerance
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.waypoints = waypoints_xyz
        self.current_index = 0
        self.finished = len(waypoints_xyz) == 0

        self.prim = self.stage.GetPrimAtPath(self.prim_path)
        if not self.prim or not self.prim.IsValid():
            raise RuntimeError(f"XformWaypointController prim not found: {self.prim_path}")
        self.xform = UsdGeom.XformCommonAPI(self.prim)

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

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def get_xyz(self) -> tuple[float, float, float]:
        x, y, z, _ = self._read_pose()
        return x, y, z

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

        if abs(yaw_error) > self.yaw_tolerance_rad:
            yaw_step = max(-self.turn_speed_rad * dt, min(self.turn_speed_rad * dt, yaw_error))
            self._write_pose(x, y, z, yaw + yaw_step)
            return

        step = min(self.move_speed * dt, planar_distance)
        step_scale = step / max(planar_distance, 1.0e-6)
        nx = x + dx * step_scale
        ny = y + dy * step_scale
        self._write_pose(nx, ny, z, desired_yaw)


def setup_camera() -> FloatingCamera:
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=[-5.46, -1.28, 0.0],
        start_orientation=[0.687852177796288, 0.0, 0.0, -0.7258507983745032],
        camera_height=1.3,
    )
    camera.init_manual()
    camera.reset()
    return camera


def ensure_isaac_asset_root() -> str:
    settings = carb.settings.get_settings()
    asset_root = settings.get("/persistent/isaac/asset_root/cloud")
    if asset_root:
        return str(asset_root).rstrip("/")

    settings.set("/persistent/isaac/asset_root/cloud", DEFAULT_ISAAC_ASSET_ROOT)
    print(
        "[Config] /persistent/isaac/asset_root/cloud was empty; "
        f"set to '{DEFAULT_ISAAC_ASSET_ROOT}'."
    )
    return DEFAULT_ISAAC_ASSET_ROOT


def load_house_usd(env_index: int, usd_path: str, usd_root_prim: str) -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    house_prim_path = f"/World/envs/env_{env_index}/House"
    house_prim = stage.DefinePrim(house_prim_path, "Xform")
    house_prim.GetReferences().AddReference(usd_path, usd_root_prim)
    print(f"[OK] House referenced at {house_prim_path} from {usd_path} ({usd_root_prim})")


def spawn_human_usd(stage: Usd.Stage) -> str:
    # Control prim (clean transform stack) so we can drive pose without touching USD-authored xform op types.
    control_prim = stage.DefinePrim(HUMAN_PRIM_PATH, "Xform")
    model_prim = stage.DefinePrim(HUMAN_MODEL_PRIM_PATH, "Xform")
    model_prim.GetReferences().AddReference(HUMAN_USD_PATH)

    xform = UsdGeom.XformCommonAPI(control_prim)
    xform.SetTranslate(Gf.Vec3d(*HUMAN_SPAWN_XYZ))
    xform.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    print(f"[Human] spawned USD at {HUMAN_MODEL_PRIM_PATH}: {HUMAN_USD_PATH}")
    print(f"[Human] spawn={HUMAN_SPAWN_XYZ}, goal={HUMAN_GOAL_XYZ}")
    return HUMAN_PRIM_PATH


def get_robot_root_xyz(robot) -> tuple[float, float, float]:
    root_pos = robot.data.root_pos_w[0]
    return tuple(float(v) for v in root_pos.tolist())


def step_world(sim: sim_utils.SimulationContext, scene: InteractiveScene, camera: FloatingCamera | None) -> None:
    dt = sim.get_physics_dt()
    sim.step(render=True)
    if camera is not None:
        camera.run(dt)
    scene.update(dt)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0, 0, 1.2], target=[0.0, 0.0, 0.0])

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")

    ensure_isaac_asset_root()
    from world.scene_cfg import HomeSceneCfg

    load_house_usd(env_index=0, usd_path=HOUSE_USD_PATH, usd_root_prim=HOUSE_USD_ROOT_PRIM)
    camera = None if args.headless else setup_camera()

    scene = InteractiveScene(HomeSceneCfg(num_envs=1, env_spacing=2.5))
    robot = scene["g1"]

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    planner = FreemapPlanner(args.freemap_path, safety_margin_m=args.path_clearance)
    robot_start_xyz = get_robot_root_xyz(robot)
    robot_waypoints = plan_waypoint_path(planner=planner, start_xyz=robot_start_xyz, goals_xyz=[ROBOT_GOAL])

    robot_controller = WaypointController(
        robot=robot,
        waypoints_xyz=robot_waypoints,
        move_speed=args.move_speed,
        turn_speed_rad=math.radians(args.turn_speed_deg),
        position_tolerance=args.position_tolerance,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
    )
    robot_controller.initialize()
    robot_controller.set_waypoints(robot_waypoints)

    if args.plan_debug_vis:
        add_path_debug_vis(stage, [robot_start_xyz] + robot_waypoints)

    human_prim_path = spawn_human_usd(stage)
    human_controller = XformWaypointController(
        stage=stage,
        prim_path=human_prim_path,
        waypoints_xyz=[HUMAN_GOAL_XYZ],
        move_speed=HUMAN_MOVE_SPEED,
        turn_speed_rad=HUMAN_TURN_SPEED_RAD,
        position_tolerance=args.position_tolerance,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
    )

    print("[OK] simplified scene ready: robot + plain human USD (no IRA).")

    step_counter = 0
    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        robot_controller.update(dt)
        human_controller.update(dt)
        step_world(sim, scene, camera)
        step_counter += 1

        if step_counter % STATUS_PRINT_EVERY_STEPS == 0:
            print(
                f"[Status] step={step_counter}, robot={get_robot_root_xyz(robot)}, "
                f"human={human_controller.get_xyz()}"
            )

        if robot_controller.finished and human_controller.finished:
            break

        if step_counter >= MAX_RUNTIME_STEPS:
            print(f"[Warn] stop after {MAX_RUNTIME_STEPS} steps before both reached goals.")
            break

    print(
        f"[Done] robot_finished={robot_controller.finished}, "
        f"human_finished={human_controller.finished}"
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
