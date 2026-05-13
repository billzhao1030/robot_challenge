import argparse
from isaaclab.app import AppLauncher

# Launch the app
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG  # Pre-built G1 config

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )
    # Unitree G1 robot
    g1: ArticulationCfg = G1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

def run_simulator(sim, scene):
    robot = scene["g1"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset every 300 steps
        if count % 300 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 0.0, 3.2], target=[0.0, 0.0, 0.5])

    scene_cfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    run_simulator(sim, scene)
    simulation_app.close()

if __name__ == "__main__":
    main()