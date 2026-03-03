import argparse
import time
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.anim.people", True)
ext_manager.set_extension_enabled_immediate("omni.replicator.agent", True)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG
from camera.floating_camera import FloatingCamera

import omni.usd
from pxr import UsdGeom, Gf

try:
    from omni.replicator.agent.core.agent_controller import AgentController
except ImportError:
    AgentController = None

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
            pos=(0.0, 0.0, 0.7),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

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
    stage = omni.usd.get_context().get_stage()
    house_prim_path = f"/World/envs/env_{env_index}/House"
    house_prim = stage.DefinePrim(house_prim_path, "Xform")
    house_prim.GetReferences().AddReference(usd_path, usd_root_prim)

def setup_ira_humans(stage):
    human_configs = [
        {"name": "Dad", "url": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Characters/Reallusion/Male_Adult_01/Male_Adult_01.usd", "pos": (1.5, 0.0, 0.0)},
        {"name": "Mom", "url": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Characters/Reallusion/Female_Adult_01/Female_Adult_01.usd", "pos": (1.5, 1.0, 0.0)},
        {"name": "Child", "url": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Characters/Reallusion/Worker/Worker.usd", "pos": (0.5, 0.5, 0.0)}
    ]

    for config in human_configs:
        prim_path = f"/World/envs/env_0/Humans/{config['name']}"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(config['url'])
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*config['pos']))

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[-2.022355307502785, 1.617730767355105, 1.2], target=[0.0, 0.0, 0.0])

    stage = omni.usd.get_context().get_stage()
    usd_path = "/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
    load_house_usd(env_index=0, usd_path=usd_path, usd_root_prim="/Root")

    if not args.headless:
        camera = setup_camera()
    else:
        camera = None

    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Humans")

    scene_cfg = HomeSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    setup_ira_humans(stage)
    sim.reset()

    if AgentController is not None:
        controller = AgentController()
        controller.command_go_to("/World/envs/env_0/Humans/Dad", (3.0, 2.0, 0.0))
        controller.command_go_to("/World/envs/env_0/Humans/Mom", (-1.0, 1.5, 0.0))
        controller.command_go_to("/World/envs/env_0/Humans/Child", (0.0, -2.0, 0.0))

    while simulation_app.is_running():
        sim.step(render=True)
        dt = sim.get_physics_dt()

        if camera is not None:
            camera.run(dt)
        scene.update(dt)

    simulation_app.close()

if __name__ == "__main__":
    main()