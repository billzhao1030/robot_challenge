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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG
from pxr import Gf, Usd, UsdGeom

from camera.floating_camera import FloatingCamera
from controllers.waypoint_controller import WaypointController
from navigation.freemap_planner import FreemapPlanner, add_path_debug_vis, plan_waypoint_path


DEFAULT_WAYPOINT: tuple[float, float, float] = (-4.75, -3.08, 0.74)
TARGET_LOCATION: tuple[float, float, float] = (-0.65, 2.14, 0.74)

LEFT_HAND_BODY_NAME = "left_six_link"
ROBOT_PRIM_PATH = "/World/envs/env_0/Robot"
MONITORED_OBJECT_PRIM_PATH = "/World/envs/env_0/House/Meshes/studyroom_767841/ornament_0015"
PICK_DISTANCE_THRESHOLD = 1
PLACE_DISTANCE_THRESHOLD = 1
HAND_ATTACH_OFFSET = Gf.Vec3d(0.0, 0.0, -0.08)
GRABBED_OBJECT_USD_PATH = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/Meshes/ornament_0015.usd"


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
            pos=(-5.51, -3.11, 0.74),
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
        start_location=[-5.46, -1.28, 0.0],
        start_orientation=[0.687852177796288, 0.0, 0.0, -0.7258507983745032],
        camera_height=1.5,
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


def get_world_translation(prim: Usd.Prim) -> Gf.Vec3d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()


def get_world_orientation(prim: Usd.Prim) -> Gf.Quatd:
    return Gf.Quatd(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotationQuat())


def set_prim_world_pose(
    prim: Usd.Prim,
    world_translation: Gf.Vec3d,
    world_orientation: Gf.Quatd,
) -> None:
    parent_prim = prim.GetParent()
    if parent_prim and parent_prim.IsValid() and parent_prim.IsA(UsdGeom.Xformable):
        parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    else:
        parent_world = Gf.Matrix4d(1.0)

    target_world = Gf.Matrix4d(1.0)
    target_world.SetRotate(world_orientation)
    target_world.SetTranslateOnly(world_translation)

    local_matrix = target_world * parent_world.GetInverse()
    local_matrix.Orthonormalize()
    xformable = UsdGeom.Xformable(prim)

    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]

    translate_op = (
        translate_ops[0]
        if translate_ops
        else xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )
    orient_op = (
        orient_ops[0]
        if orient_ops
        else xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )

    translate_op.Set(local_matrix.ExtractTranslation())
    local_quat = local_matrix.ExtractRotationQuat()
    orient_attr = orient_op.GetAttr()
    if orient_attr.GetTypeName() == "quatf":
        orient_op.Set(Gf.Quatf(local_quat))
    else:
        orient_op.Set(Gf.Quatd(local_quat))


def set_prim_local_pose(
    prim: Usd.Prim,
    local_translation: Gf.Vec3d,
    local_orientation: Gf.Quatd,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]

    translate_op = (
        translate_ops[0]
        if translate_ops
        else xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )
    orient_op = (
        orient_ops[0]
        if orient_ops
        else xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )

    translate_op.Set(local_translation)
    orient_attr = orient_op.GetAttr()
    if orient_attr.GetTypeName() == "quatf":
        orient_op.Set(Gf.Quatf(local_orientation))
    else:
        orient_op.Set(Gf.Quatd(local_orientation))


def vec3d_to_xyz(value: Gf.Vec3d) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def get_robot_root_xyz(robot) -> tuple[float, float, float]:
    root_pos = robot.data.root_pos_w[0]
    return tuple(float(v) for v in root_pos.tolist())


def distance_xyz(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    return math.dist(point_a, point_b)


def plan_phase_waypoints(
    planner: FreemapPlanner,
    start_xyz: tuple[float, float, float],
    goal_xyz: tuple[float, float, float],
    approach_threshold: float | None = None,
) -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    approach_goal = goal_xyz
    if approach_threshold is not None:
        free_goal_xy = planner.find_free_point_within_radius(goal_xyz[0], goal_xyz[1], approach_threshold)
        if free_goal_xy is not None:
            approach_goal = (free_goal_xy[0], free_goal_xy[1], goal_xyz[2])
        else:
            fallback_goal = planner.find_nearest_reachable(goal_xyz[0], goal_xyz[1])
            if fallback_goal is None:
                raise RuntimeError(f"Could not find a reachable approach point near {goal_xyz}.")
            approach_goal = (fallback_goal[0], fallback_goal[1], goal_xyz[2])
            print(
                f"[Waypoint] no reachable point inside {approach_threshold:.2f} m of {goal_xyz}; "
                f"falling back to nearest reachable {approach_goal}."
            )

    return plan_waypoint_path(planner=planner, start_xyz=start_xyz, goals_xyz=[approach_goal]), approach_goal


class ProximityGrabber:
    def __init__(
        self,
        stage,
        robot,
        hand_body_name: str,
        object_prim_path: str,
        trigger_distance: float,
        grabbed_object_usd_path: str,
    ) -> None:
        self.stage = stage
        self.robot = robot
        self.object_prim_path = object_prim_path
        self.object_prim = stage.GetPrimAtPath(object_prim_path)
        self.trigger_distance = trigger_distance
        self.is_attached = False
        self.is_released = False
        self._step_count = 0
        self.grabbed_object_usd_path = grabbed_object_usd_path
        self.attached_object_prim_path: str | None = None
        self.original_world_orientation = get_world_orientation(self.object_prim)
        hand_body_ids, hand_body_names = robot.find_bodies(hand_body_name)
        if len(hand_body_ids) == 0:
            raise RuntimeError(f"Hand body not found: {hand_body_name}")
        self.hand_body_idx = hand_body_ids[0]
        self.hand_body_name = hand_body_names[0]

        if not self.object_prim or not self.object_prim.IsValid():
            raise RuntimeError(f"Object prim not found: {object_prim_path}")

    def distance_to_object(self) -> float:
        robot_root = self.robot.data.root_pos_w[0].clone()
        robot_probe = Gf.Vec3d(float(robot_root[0]), float(robot_root[1]), 0.8)
        object_pos = get_world_translation(self.object_prim)
        return (
            (robot_probe[0] - object_pos[0]) ** 2
            + (robot_probe[1] - object_pos[1]) ** 2
            + (robot_probe[2] - object_pos[2]) ** 2
        ) ** 0.5

    def _attach_object_under_hand(self) -> None:
        object_name = self.object_prim.GetName()
        hand_parent_path = f"{ROBOT_PRIM_PATH}/{self.hand_body_name}"
        attached_object_path = f"{hand_parent_path}/{object_name}"

        self.stage.RemovePrim(attached_object_path)
        original_override = self.stage.OverridePrim(self.object_prim_path)
        original_override.SetActive(False)

        attached_prim = self.stage.DefinePrim(attached_object_path, "Xform")
        attached_prim.GetReferences().AddReference(self.grabbed_object_usd_path)
        set_prim_local_pose(
            attached_prim,
            local_translation=HAND_ATTACH_OFFSET,
            local_orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        )

        self.object_prim = attached_prim
        self.attached_object_prim_path = attached_object_path

    def release_to_world(self, release_xyz: tuple[float, float, float]) -> None:
        if not self.is_attached or self.is_released:
            return

        if self.attached_object_prim_path is not None:
            self.stage.RemovePrim(self.attached_object_prim_path)

        restored_prim = self.stage.OverridePrim(self.object_prim_path)
        restored_prim.SetActive(True)
        references = restored_prim.GetReferences()
        references.ClearReferences()
        references.AddReference(self.grabbed_object_usd_path)
        set_prim_world_pose(
            restored_prim,
            world_translation=Gf.Vec3d(*release_xyz),
            world_orientation=self.original_world_orientation,
        )

        self.object_prim = restored_prim
        self.is_released = True
        print(f"[Grab] released {self.object_prim_path} to {list(release_xyz)}")

    def update(self) -> None:
        self._step_count += 1
        if not self.is_attached and not self.is_released:
            distance = self.distance_to_object()
            if self._step_count % 120 == 0:
                print(f"[Grab] distance to object: {distance:.3f} m")
            if distance < self.trigger_distance:
                self.is_attached = True
                self._attach_object_under_hand()
                print(
                    f"[Grab] attached {self.object_prim.GetName()} under {ROBOT_PRIM_PATH}/{self.hand_body_name} at distance {distance:.3f} m"
                )

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
            target_distance = distance_xyz(robot_xyz, TARGET_LOCATION)
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
