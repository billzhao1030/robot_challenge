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

import carb
import numpy as np
from isaaclab.app import AppLauncher

# -------------------------
# 1) Parse args + launch app
# -------------------------
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--house_usd", type=str, default="", help="Absolute path to house usd")
parser.add_argument("--house_root_prim", type=str, default="/Root",
                    help="Root prim inside the USD (demo.py uses /Root)")
parser.add_argument("--camera_height", default=1.5, type=float, help="Initial camera height")
parser.add_argument("--camera_move_speed", default=1.0, type=float)
parser.add_argument("--camera_turn_speed", default=30.0, type=float)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------
# 2) Import Isaac Lab AFTER launching
# -------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG

from camera.floating_camera import FloatingCameraController

@configclass
class MySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    g1: ArticulationCfg = G1_CFG.replace(
        prim_path="/World/envs/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.9),  # adjust after you see house origin
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # init_state is the supported way to set spawn pose. :contentReference[oaicite:2]{index=2}

    human_0 = RigidObjectCfg(
        prim_path="/World/envs/Humans/Human_0",
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
        prim_path="/World/envs/Humans/Human_1",
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
        prim_path="/World/envs/Humans/Human_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.14,
            height=1.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.5, 0.55]),
    )

def load_house_usd(env_index: int, usd_path: str, usd_root_prim: str) -> None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    house_prim_path = "/World/House"
    house_prim = stage.DefinePrim(house_prim_path, "Xform")

    # demo.py-style: reference a specific prim inside the file (often /Root)
    house_prim.GetReferences().AddReference(usd_path, usd_root_prim)
    print(f"[OK] House referenced at {house_prim_path} from {usd_path} ({usd_root_prim})")

def run(sim_seconds: int = 120) -> None:
    # --- Build modules ---
    ws = WorldState.default()

    sim = SimInterfaceStub()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mem_dir = os.path.join("memories", run_id)
    mem = MemoryWriter(out_dir=mem_dir)

    humans = [
        HumanAgent("Dad", request_probability=0.25, move_probability=0.15, memory=mem),
        HumanAgent("Mom", request_probability=0.18, move_probability=0.10, memory=mem),
        HumanAgent("Child", request_probability=0.10, move_probability=0.25, memory=mem),
    ]

    brain = RobotBrain(low_battery_threshold=0.25)
    executor = RobotExecutor(sim=sim)

    # --- Time scales ---
    physics_hz = 120.0
    physics_dt = 1.0 / physics_hz

    decision_hz = 5.0
    decision_dt = 1.0 / decision_hz
    next_decision_t = 0.0

    # --- Loop ---
    wall_start = time.time()
    logical_t = 0.0

    ws.log_event("=== Simulation start ===")

    while logical_t < sim_seconds:
        # Physics step (fast)
        sim.step(physics_dt)
        logical_t += physics_dt

        # Decision step (slow)
        if logical_t >= next_decision_t:
            # Advance logical clock by 1 second for world time display
            ws.advance_time(1)

            # Humans update world state (logical)
            for h in humans:
                h.update(ws)

            # Robot decides symbolic command
            cmd = brain.decide(ws)

            # Execute symbolic command into low-level sim calls
            if cmd is not None:
                executor.execute(cmd, ws)

            # Print a compact status line
            robot = ws.data["robot"]
            print(
                f"[{ws.data['time_str']}] "
                f"Robot@{robot['location']} bat={robot['battery']:.2f} holding={robot['holding']} "
                f"reqs={len(ws.data['requests'])} last={robot['last_command']}"
            )

            next_decision_t += decision_dt

        # Optional: throttle wall-clock so it doesn't run instantly
        # (Set to 0.0 to run as fast as possible)
        time.sleep(0.0)

    ws.log_event("=== Simulation end ===")

    # Dump last N events
    print("\n--- Recent Events ---")
    for e in ws.data["events"][-25:]:
        print(f"{e['t']}: {e['msg']}")
        

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 0.0, 2.5], target=[0.0, 0.0, 0.8])

    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()

    # Create parent prims
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/Humans")

    scene_cfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Load house USD AFTER reset so /World/envs/env_0 exists cleanly
    usd_path = "/home/xunyi/isaacsim4.5/projects/robot_challenge/data/kujiale_0003/kujiale_0003.usda"
    load_house_usd(env_index=0, usd_path=usd_path, usd_root_prim="/Root")

    print("[OK] Stage ready: house + G1 + 3 humans. Close window to exit.")

    loc = [-2.43, 1.61, 0.0]
    ori = [0.69, 0.0, 0.0, -0.73]
    camera_controller = FloatingCameraController(
        simulation_app=simulation_app,
        start_location=loc,
        start_orientation=ori,
        camera_height=args_cli.camera_height,
        move_speed=args_cli.camera_move_speed,
        turn_speed=args_cli.camera_turn_speed,
    )
    camera_controller.init_manual()
    camera_controller.reset()

    while simulation_app.is_running():
        sim.step(render=True)
        dt = sim.get_physics_dt()
        camera_controller.run(dt)
        scene.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     run(sim_seconds=180)