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


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[-5.46, -1.28, 1.2],
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

    default_root_state = robot.data.default_root_state[0].clone()
    start_xyz = tuple(float(v) for v in default_root_state[:3].tolist())
    planner = FreemapPlanner(args.freemap_path, safety_margin_m=args.path_clearance)
    object_location = vec3d_to_xyz(get_world_translation(stage.GetPrimAtPath(MONITORED_OBJECT_PRIM_PATH)))
    planned_waypoints, default_approach = plan_phase_waypoints(
        planner=planner,
        start_xyz=start_xyz,
        goal_xyz=DEFAULT_WAYPOINT,
    )

    controller = WaypointController(
        robot=robot,
        waypoints_xyz=planned_waypoints,
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

    def load_route(
        start_pose_xyz: tuple[float, float, float],
        goal_xyz: tuple[float, float, float],
        label: str,
        approach_threshold: float | None = None,
    ) -> tuple[float, float, float]:
        route_waypoints, approach_goal = plan_phase_waypoints(
            planner=planner,
            start_xyz=start_pose_xyz,
            goal_xyz=goal_xyz,
            approach_threshold=approach_threshold,
        )
        controller.set_waypoints(route_waypoints)
        if args.plan_debug_vis:
            add_path_debug_vis(stage, [start_pose_xyz] + route_waypoints)
        print(f"[Waypoint] {label} goal: {goal_xyz}")
        print(f"[Waypoint] {label} approach goal: {approach_goal}")
        print(f"[Waypoint] {label} planned path: {controller.waypoints.tolist()}")
        return approach_goal

    if args.plan_debug_vis:
        add_path_debug_vis(stage, [start_xyz] + planned_waypoints)

    print("[OK] Scene ready: house + G1.")
    print(f"[Waypoint] default waypoint: {DEFAULT_WAYPOINT}")
    print(f"[Task] object location: {object_location}")
    print(f"[Task] target location: {TARGET_LOCATION}")
    print(f"[Waypoint] freemap resolution: {planner.grid_resolution:.4f} m, safety margin: {args.path_clearance:.3f} m")
    print(f"[Waypoint] default approach goal: {default_approach}")
    print(f"[Waypoint] default planned path: {controller.waypoints.tolist()}")
    print(f"[Grab] monitoring {MONITORED_OBJECT_PRIM_PATH}")
    task_phase = "to_default"
    route_blocked_logged = False

    while simulation_app.is_running():
        dt = sim.get_physics_dt()

        robot_xyz = get_robot_root_xyz(robot)

        if task_phase == "to_default" and controller.finished:
            load_route(
                start_pose_xyz=robot_xyz,
                goal_xyz=object_location,
                label="object",
                approach_threshold=PICK_DISTANCE_THRESHOLD,
            )
            task_phase = "to_object"
            route_blocked_logged = False

        if task_phase == "to_object" and not grabber.is_attached:
            object_distance = grabber.distance_to_object()
            if object_distance <= PICK_DISTANCE_THRESHOLD:
                controller.set_waypoints([])
                task_phase = "grab_object"
                print(f"[Task] within pick threshold ({object_distance:.3f} m). Grabbing object.")

        if task_phase == "to_target" and not grabber.is_released:
            target_distance = planar_distance_xy(robot_xyz, TARGET_LOCATION)
            if target_distance <= PLACE_DISTANCE_THRESHOLD:
                controller.set_waypoints([])
                grabber.release_to_world(TARGET_LOCATION)
                task_phase = "done"
                print(f"[Task] within place threshold ({target_distance:.3f} m). Object placed.")
            elif controller.finished and not route_blocked_logged:
                print(
                    f"[Task] route to target ended at {robot_xyz}, which is still "
                    f"{target_distance:.3f} m from the target."
                )
                route_blocked_logged = True

        controller.update(dt)
        grabber.update()

        if task_phase in {"to_object", "grab_object"} and grabber.is_attached:
            robot_xyz = get_robot_root_xyz(robot)
            load_route(
                start_pose_xyz=robot_xyz,
                goal_xyz=TARGET_LOCATION,
                label="target",
                approach_threshold=PLACE_DISTANCE_THRESHOLD,
            )
            task_phase = "to_target"
            route_blocked_logged = False

        sim.step(render=True)
        if camera is not None:
            camera.run(dt)
        scene.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
