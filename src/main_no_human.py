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

    print("[OK] Physical stage ready: house + G1 + 3 humans. Close window to exit.")

    # --- Build your "whole-day" modules (from run()) ---
    ws = WorldState.default()

    # IMPORTANT: replace stub with a real sim adapter that talks to Isaac,
    # or keep stub for now if you only print logic.
    sim_if = SimInterfaceStub()  # later: SimInterfaceIsaac(sim, scene, stage)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mem_dir = os.path.join(os.getcwd(), "memories", run_id)
    mem = MemoryWriter(out_dir=mem_dir)

    humans = [
        HumanAgent("Dad", request_probability=0.25, move_probability=0.15, memory=mem),
        HumanAgent("Mom", request_probability=0.18, move_probability=0.10, memory=mem),
        HumanAgent("Child", request_probability=0.10, move_probability=0.25, memory=mem),
    ]

    brain = RobotBrain(low_battery_threshold=0.25)
    executor = RobotExecutor(sim=sim_if)

    # --- Time scales ---
    decision_hz = 0.5
    decision_dt = 1.0 / decision_hz

    sim_seconds = 24 * 60 * 60  # for a day
    sim_t = 0.0
    decision_accum = 0.0

    ws.log_event("=== Simulation start ===")


    while simulation_app.is_running() and sim_t < sim_seconds:
        sim.step(render=True)
        dt = sim.get_physics_dt()

        sim_t += dt

        # keep scene/camera updated each physics frame
        if camera is not None:
            camera.run(dt)
        scene.update(dt)

        decision_accum += dt
        if decision_accum >= decision_dt:
            # if dt jitters, you can do while-loop to "catch up" cleanly
            while decision_accum >= decision_dt:
                decision_accum -= decision_dt

                # Option A: advance world time by decision_dt seconds (recommended)
                ws.advance_time(int(decision_dt) * 60)

                # (optional) sync from Isaac -> ws here
                # sim_if.sync_worldstate_from_sim(ws)

                for h in humans:
                    h.update(ws)

                cmd = brain.decide(ws)
                if cmd is not None:
                    executor.execute(cmd, ws)

                robot = ws.data["robot"]
                print(
                    f"[{ws.data['time_str']}] "
                    f"Robot@{robot['location']} bat={robot['battery']:.2f} "
                    f"holding={robot['holding']} reqs={len(ws.data['requests'])} "
                    f"last={robot['last_command']}"
                )

    ws.log_event("=== Simulation end ===")

    print("\n--- Recent Events ---")
    for e in ws.data["events"][-25:]:
        print(f"{e['t']}: {e['msg']}")

    simulation_app.close()

if __name__ == "__main__":
    main()