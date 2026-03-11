from __future__ import annotations

import math
from dataclasses import dataclass

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
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from pxr import Gf

from camera.floating_camera import FloatingCamera
from controllers.waypoint_controller import WaypointController
from navigation.freemap_planner import FreemapPlanner, add_path_debug_vis, plan_waypoint_path
from world.grabber import ProximityGrabber, get_world_translation, vec3d_to_xyz
from world.scene_cfg import HomeSceneCfg


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
EXAMPLE_START_AFTER_SECONDS = 2.0
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
    runtime.sim.step(render=True)
    if runtime.camera is not None:
        runtime.camera.run(dt)
    runtime.scene.update(dt)


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

    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")

    usd_path = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
    load_house_usd(env_index=0, usd_path=usd_path, usd_root_prim="/Root")

    camera = None if args.headless else setup_camera()

    scene = InteractiveScene(HomeSceneCfg(num_envs=1, env_spacing=2.5))
    robot = scene["g1"]

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    planner = FreemapPlanner(args.freemap_path, safety_margin_m=args.path_clearance)
    object_location = (
        PICK_OBJECT_LOCATION
        if PICK_OBJECT_LOCATION is not None
        else vec3d_to_xyz(get_world_translation(stage.GetPrimAtPath(MONITORED_OBJECT_PRIM_PATH)))
    )
    target_location = PICK_TARGET_LOCATION

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
    )

    print("[OK] Scene ready: house + G1.")
    print(f"[Waypoint] default waypoint: {DEFAULT_WAYPOINT}")
    print(f"[Task] object location: {object_location}")
    print(f"[Task] target location: {target_location}")
    print(f"[Waypoint] freemap resolution: {planner.grid_resolution:.4f} m, safety margin: {args.path_clearance:.3f} m")
    print(f"[Grab] monitoring {MONITORED_OBJECT_PRIM_PATH}")

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