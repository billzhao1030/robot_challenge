from __future__ import annotations
import os
import argparse
import time
from datetime import datetime

from world.world_state import WorldState
from sim.sim_interface import SimInterfaceStub
from agents.human_agent import HumanAgent
from agents.robot_brain import RobotBrain
from executor.robot_executor import RobotExecutor
from memory.memory_writer import MemoryWriter

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
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG

from camera.floating_camera import FloatingCamera

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
            pos=(0.0, 0.0, 0.7),  # adjust after you see house origin
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

def attach_human_visual(proxy_prim_path: str, human_usd_path: str, usd_root_prim: str = None):
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()

    # Create a child Xform under the proxy
    vis_path = f"{proxy_prim_path}/Visual"
    UsdGeom.Xform.Define(stage, vis_path)

    # Reference the human USD under this child
    prim = stage.GetPrimAtPath(vis_path)
    if usd_root_prim:
        prim.GetReferences().AddReference(human_usd_path, usd_root_prim)
    else:
        prim.GetReferences().AddReference(human_usd_path)

    print(f"[OK] Attached human visual: {human_usd_path} -> {vis_path}")

def setup_camera():
    loc = [-2.022355307502785, 1.617730767355105, 0.0]
    ori = [0.687852177796288, 0.0, 0.0, -0.7258507983745032]
    cam_height = 1.1
    camera = FloatingCamera(
        simulation_app=simulation_app,
        start_location=loc,
        start_orientation=ori,
        camera_height=cam_height
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


def main():
    # --- Isaac sim setup ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01) # 100Hz
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[-2.022355307502785, 1.617730767355105, 1.2], target=[0.0, 0.0, 0.0])

    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()

    # Load house USD
    usd_path = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
    load_house_usd(env_index=0, usd_path=usd_path, usd_root_prim="/Root")

    if not args.headless:
        camera = setup_camera()
    else:
        camera = None

    # Setup Robot and Human
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Humans")

    scene_cfg = HomeSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    human_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
    attach_human_visual("/World/envs/env_0/Humans/Human_0", human_usd, usd_root_prim="/Root")

    print("[OK] Physical stage ready: house + G1 + 3 humans. Close window to exit.")

    while simulation_app.is_running():
        sim.step(render=True)
        dt = sim.get_physics_dt()

        # keep scene/camera updated each physics frame
        if camera is not None:
            camera.run(dt)
        scene.update(dt)


    simulation_app.close()

if __name__ == "__main__":
    main()