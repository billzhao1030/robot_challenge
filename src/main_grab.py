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


DEFAULT_WAYPOINTS: list[tuple[float, float, float]] = [
    (-4.75, -3.08, 0.74),
]

LEFT_HAND_BODY_NAME = "left_six_link"
MONITORED_OBJECT_PRIM_PATH = "/World/envs/env_0/House/Meshes/studyroom_767841/ornament_0015"
GRAB_DISTANCE_THRESHOLD = 1.0
HAND_ATTACH_OFFSET = Gf.Vec3d(0.0, 0.0, -0.08)


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
            pos=(-5.11, -1.11, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


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


def get_world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def multiply_transforms(left: Gf.Matrix4d, right: Gf.Matrix4d) -> Gf.Matrix4d:
    result = Gf.Matrix4d(left)
    result *= right
    return result


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
    if orient_op.GetAttr().GetTypeName() == "quatf":
        orient_op.Set(Gf.Quatf(local_quat))
    else:
        orient_op.Set(Gf.Quatd(local_quat))


def set_prim_world_transform(prim: Usd.Prim, world_transform: Gf.Matrix4d) -> None:
    set_prim_world_pose(
        prim,
        world_translation=world_transform.ExtractTranslation(),
        world_orientation=Gf.Quatd(world_transform.ExtractRotationQuat()),
    )


class DirectProximityGrabber:
    def __init__(
        self,
        stage,
        robot,
        hand_body_name: str,
        object_prim_path: str,
        trigger_distance: float,
    ) -> None:
        self.stage = stage
        self.robot = robot
        self.object_prim = stage.GetPrimAtPath(object_prim_path)
        self.trigger_distance = trigger_distance
        self.is_attached = False
        self._step_count = 0
        self.attached_root_relatives: list[tuple[Usd.Prim, Gf.Matrix4d]] = []

        hand_body_ids, hand_body_names = robot.find_bodies(hand_body_name)
        if len(hand_body_ids) == 0:
            raise RuntimeError(f"Hand body not found: {hand_body_name}")
        self.hand_body_idx = hand_body_ids[0]
        self.hand_body_name = hand_body_names[0]

        if not self.object_prim or not self.object_prim.IsValid():
            raise RuntimeError(f"Object prim not found: {object_prim_path}")

    def _collect_attached_xforms(self) -> None:
        root_world = get_world_transform(self.object_prim)
        root_world_inv = root_world.GetInverse()
        self.attached_root_relatives = []

        for prim in Usd.PrimRange(self.object_prim):
            if not prim.IsValid() or prim == self.object_prim or not prim.IsA(UsdGeom.Xformable):
                continue
            child_world = get_world_transform(prim)
            rel_transform = multiply_transforms(root_world_inv, child_world)
            self.attached_root_relatives.append((prim, rel_transform))

        print(f"[GrabDirect] captured {len(self.attached_root_relatives)} child xformable prims under the original object")

    def update(self) -> None:
        self._step_count += 1
        if not self.is_attached:
            robot_root = self.robot.data.root_pos_w[0].clone()
            robot_probe = Gf.Vec3d(float(robot_root[0]), float(robot_root[1]), 0.8)
            object_pos = get_world_translation(self.object_prim)
            distance = (
                (robot_probe[0] - object_pos[0]) ** 2
                + (robot_probe[1] - object_pos[1]) ** 2
                + (robot_probe[2] - object_pos[2]) ** 2
            ) ** 0.5
            if self._step_count % 120 == 0:
                print(f"[GrabDirect] distance to object: {distance:.3f} m")
            if distance < self.trigger_distance:
                self.is_attached = True
                self._collect_attached_xforms()
                print(
                    f"[GrabDirect] attached original {self.object_prim.GetPath()} to {self.hand_body_name} at distance {distance:.3f} m"
                )

        if self.is_attached:
            hand_world_translation = Gf.Vec3d(*self.robot.data.body_pose_w[0, self.hand_body_idx, :3].tolist())
            hand_world_orientation = Gf.Quatd(*self.robot.data.body_quat_w[0, self.hand_body_idx].tolist())
            hand_rotation = Gf.Rotation(hand_world_orientation)
            attach_translation = hand_world_translation + hand_rotation.TransformDir(HAND_ATTACH_OFFSET)
            target_root_world = Gf.Matrix4d(1.0)
            target_root_world.SetRotate(hand_world_orientation)
            target_root_world.SetTranslateOnly(attach_translation)

            for prim, rel_transform in self.attached_root_relatives:
                set_prim_world_transform(prim, multiply_transforms(target_root_world, rel_transform))


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

    initial_goals = DEFAULT_WAYPOINTS.copy()
    default_root_state = robot.data.default_root_state[0].clone()
    start_xyz = tuple(float(v) for v in default_root_state[:3].tolist())
    planner = FreemapPlanner(args.freemap_path, safety_margin_m=args.path_clearance)
    planned_waypoints = plan_waypoint_path(
        planner=planner,
        start_xyz=start_xyz,
        goals_xyz=initial_goals,
    )

    if args.plan_debug_vis:
        add_path_debug_vis(stage, [start_xyz] + planned_waypoints)

    controller = WaypointController(
        robot=robot,
        waypoints_xyz=planned_waypoints,
        move_speed=args.move_speed,
        turn_speed_rad=math.radians(args.turn_speed_deg),
        position_tolerance=args.position_tolerance,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
    )
    controller.initialize()

    grabber = DirectProximityGrabber(
        stage=stage,
        robot=robot,
        hand_body_name=LEFT_HAND_BODY_NAME,
        object_prim_path=MONITORED_OBJECT_PRIM_PATH,
        trigger_distance=GRAB_DISTANCE_THRESHOLD,
    )

    print("[OK] Scene ready: house + G1.")
    print(f"[Waypoint] raw goals: {initial_goals}")
    print(f"[Waypoint] freemap resolution: {planner.grid_resolution:.4f} m, safety margin: {args.path_clearance:.3f} m")
    print(f"[Waypoint] planned path: {controller.waypoints.tolist()}")
    print(f"[GrabDirect] monitoring original {MONITORED_OBJECT_PRIM_PATH}")

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        controller.update(dt)
        grabber.update()
        sim.step(render=True)
        if camera is not None:
            camera.run(dt)
        scene.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
