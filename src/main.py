from __future__ import annotations
import os
import time
from datetime import datetime

from world.world_state import WorldState
from sim.sim_interface import SimInterfaceStub
from agents.human_agent import HumanAgent
from agents.robot_brain import RobotBrain
from executor.robot_executor import RobotExecutor
from memory.memory_writer import MemoryWriter


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


if __name__ == "__main__":
    run(sim_seconds=180)