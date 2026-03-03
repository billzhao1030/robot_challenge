from __future__ import annotations

import argparse
import math
from datetime import datetime

from isaaclab.app import AppLauncher

# -------------------------
# 1) Parse args + launch app
# -------------------------
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -------------------------
# 2) Import Isaac Lab AFTER launching
# -------------------------
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets import G1_CFG

from camera.floating_camera import FloatingCamera


# -------------------------
# Scene config
# -------------------------
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
            pos=(0.0, 0.0, 0.8),  # keep consistent with your G1 init height
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    human_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Humans/Human_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.18,
            height=1.75,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 0.0, 0.875]),
    )

    human_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Humans/Human_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.17,
            height=1.62,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 1.0, 0.81]),
    )

    human_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Humans/Human_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.14,
            height=1.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.5, 0.55]),
    )


# -------------------------
# Helpers: camera, house loading, slow teleport
# -------------------------
def setup_camera():
    loc = [-2.022355307502785, 1.617730767355105, 0.0]
    ori = [0.687852177796288, 0.0, 0.0, -0.7258507983745032]
    cam_height = 1.1
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=loc,
        start_orientation=ori,
        camera_height=cam_height,
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


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """yaw: (N,) -> quat wxyz: (N,4)"""
    half = 0.5 * yaw
    return torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
        dim=-1,
    )


def slow_teleport_step(
    robot,
    target_xy: tuple[float, float],
    dt: float,
    speed: float = 0.25,
    face_motion: bool = True,
    fixed_z: float | None = None,
    eps: float = 1e-3,
) -> bool:
    """
    Moves robot root pose in small increments each physics step.
    - target_xy in world coordinates
    - speed in meters/sec
    Returns True if reached.
    """
    device = robot.device
    root = robot.data.root_state_w  # (num_envs, 13) = pos(3), quat(4), linvel(3), angvel(3)
    pos = root[:, 0:3]
    quat = root[:, 3:7]

    target_xy_t = torch.tensor(target_xy, device=device, dtype=pos.dtype).view(1, 2).repeat(pos.shape[0], 1)
    cur_xy = pos[:, 0:2]

    delta = target_xy_t - cur_xy
    dist = torch.norm(delta, dim=-1, keepdim=True)

    # close enough -> snap
    if torch.all(dist < eps):
        new_pos = pos.clone()
        new_pos[:, 0:2] = target_xy_t
        if fixed_z is not None:
            new_pos[:, 2] = fixed_z
        robot.write_root_pose_to_sim(torch.cat([new_pos, quat], dim=-1))
        return True

    max_step = speed * dt
    step = delta / (dist + 1e-6) * torch.minimum(
        dist, torch.tensor([[max_step]], device=device, dtype=pos.dtype)
    )
    new_xy = cur_xy + step

    new_pos = pos.clone()
    new_pos[:, 0:2] = new_xy
    if fixed_z is not None:
        new_pos[:, 2] = fixed_z

    new_quat = quat
    if face_motion:
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        new_quat = yaw_to_quat_wxyz(yaw)

    robot.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1))
    return False


# -------------------------
# Main
# -------------------------
def main():
    # --- Isaac Lab simulation context ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)  # 100 Hz
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[-2.022355307502785, 1.617730767355105, 1.2],
        target=[0.0, 0.0, 0.0],
    )

    # Stage handle + prim creation (IMPORTANT: create env prims before spawning scene assets)
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()

    # Create env prims so env_.*/Humans paths exist
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Humans")

    # Load house USD reference under env_0
    usd_path = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
    load_house_usd(env_index=0, usd_path=usd_path, usd_root_prim="/Root")

    # Optional floating camera
    camera = None
    if not args.headless:
        camera = setup_camera()

    # Build scene (spawns ground/light/robot/humans)
    scene_cfg = HomeSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Grab robot handle
    robot = scene["g1"]

    # Reset sim & scene
    sim.reset()
    scene.reset()

    print("[OK] Physical stage ready: house + G1 + 3 humans. Close window to exit.")

    # Time
    sim_t = 0.0
    sim_seconds = 60.0

    # Slow-teleport “walking path” (world XY)
    waypoints = [
        (-0.5, 0.0),
        (-1.0, 0.5),
        (-1.5, 1.0),
        (-2.0, 2.0),
    ]
    wp_i = 0

    # Keep Z stable (match G1 init height)
    fixed_z = 0.74

    while simulation_app.is_running() and sim_t < sim_seconds:
        sim.step(render=True)
        dt = sim.get_physics_dt()
        sim_t += dt

        # ---- slow teleport step for the robot ----
        if wp_i < len(waypoints):
            reached = slow_teleport_step(
                robot=robot,
                target_xy=waypoints[wp_i],
                dt=dt,
                speed=0.5,        # <- lower this if you see jitter or collisions snagging
                face_motion=True,  # face toward waypoint
                fixed_z=fixed_z,
            )
            if reached:
                wp_i += 1

        # update camera + scene each frame
        if camera is not None:
            camera.run(dt)
        scene.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()