#!/usr/bin/env python3
"""
three_people_sim.py
====================
Isaac Sim 4.5 standalone script that spawns 3 animated people in a warehouse
and drives them through a sequence of actions: Walk (GoTo), Idle, LookAround.

Usage (from your Isaac Sim install root):
  Linux:   ./python.sh /path/to/three_people_sim/three_people_sim.py
  Windows: .\python.bat \path\to\three_people_sim\three_people_sim.py

The script will:
  1. Launch the Isaac Sim app in headless/windowed mode.
  2. Load a simple warehouse scene.
  3. Enable the IRA (Isaacsim.Replicator.Agent) extensions.
  4. Spawn 3 characters and assign them a hand-crafted command sequence
     that covers GoTo (walk), Idle, and LookAround.
  5. Run the simulation for ~300 frames (~10 s @ 30 fps) then exit.

Adjustable constants are grouped at the top of the file.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import time
import argparse
import tempfile

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap  (must happen before any omni.* imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

# ── Tweak these to taste ────────────────────────────────────────────────────
HEADLESS         = False          # True = no window (useful for servers)
SIMULATION_FRAMES = 300           # frames to run  (~10 s at 30 fps)
SCENE_USD        = (
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Environments/"
    "Simple_Warehouse/full_warehouse.usd"
)
# Fallback: use an empty stage if the Nucleus asset is unavailable
FALLBACK_EMPTY_STAGE = True

CHARACTER_ASSET_PATH = (
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/People/Characters/"
)
# ────────────────────────────────────────────────────────────────────────────

# Start the app BEFORE importing omni packages
launch_config = {
    "headless": HEADLESS,
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(launch_config)

# ---------------------------------------------------------------------------
# Omniverse / Isaac Sim imports  (safe after SimulationApp is up)
# ---------------------------------------------------------------------------
import carb
import omni.kit.app
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from pxr import UsdGeom, Gf, Sdf
import omni.kit.commands

# ---------------------------------------------------------------------------
# Enable required extensions
# ---------------------------------------------------------------------------
print("[three_people_sim] Enabling required extensions …")
enable_extension("omni.anim.graph.core")
enable_extension("omni.anim.navigation.core")
enable_extension("omni.anim.people")
enable_extension("isaacsim.replicator.agent.core")

# Give Kit a couple of frames to finish loading the extensions
for _ in range(10):
    simulation_app.update()

# ---------------------------------------------------------------------------
# Helper: write the command file used by omni.anim.people / IRA
# ---------------------------------------------------------------------------
def build_command_file(path: str) -> None:
    """
    Each line:  <CharacterPrimName> <Command> [params …]

    Person_01  – walks a path, idles, looks around, walks back
    Person_02  – idles a bit, looks around, then walks
    Person_03  – looks around, then walks a long path
    """
    lines = [
        "# --- Person 01 : walk → idle → look → walk back ---",
        "Character GoTo  5.0  0.0  0  0",        # walk to (5,0,0), face 0°
        "Character GoTo  5.0  5.0  0  90",        # walk to (5,5,0), face 90°
        "Character Idle 5",                        # stand still 5 s
        "Character LookAround 4",                  # head turn 4 s
        "Character GoTo  0.0  0.0  0  180",        # walk back to origin

        "",
        "# --- Person 02 : idle → look → walk ---",
        "Character_01 Idle 3",
        "Character_01 LookAround 3",
        "Character_01 GoTo  -3.0  3.0  0  45",
        "Character_01 GoTo   3.0  6.0  0  270",
        "Character_01 Idle 4",

        "",
        "# --- Person 03 : look → walk a loop ---",
        "Character_02 LookAround 5",
        "Character_02 GoTo  2.0  -3.0  0  0",
        "Character_02 GoTo  6.0  -3.0  0  90",
        "Character_02 GoTo  6.0   3.0  0  180",
        "Character_02 GoTo  2.0   3.0  0  270",
        "Character_02 Idle 3",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[three_people_sim] Command file written → {path}")


# ---------------------------------------------------------------------------
# Helper: write the IRA YAML config
# ---------------------------------------------------------------------------
def build_config_file(config_path: str, command_file_path: str) -> None:
    scene_path = SCENE_USD if not FALLBACK_EMPTY_STAGE else ""

    content = f"""\
isaacsim.replicator.agent:
  version: 0.5.1
  global:
    seed: 42
    simulation_length: {SIMULATION_FRAMES}
  scene:
    asset_path: {scene_path}
  sensor:
    camera_num: 1
  character:
    asset_path: {CHARACTER_ASSET_PATH}
    command_file: {command_file_path}
    filters: ''
    num: 3
  robot:
    nova_carter_num: 0
    transporter_num: 0
    write_data: false
  replicator:
    writer: BasicWriter
    parameters:
      output_dir:
      rgb: true
"""
    with open(config_path, "w") as f:
        f.write(content)

    print(f"[three_people_sim] Config file written → {config_path}")


# ---------------------------------------------------------------------------
# Main simulation logic
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Three-person Isaac Sim demo")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without a viewport window"
    )
    parser.add_argument(
        "--frames", type=int, default=SIMULATION_FRAMES,
        help=f"Number of simulation frames to run (default: {SIMULATION_FRAMES})"
    )
    args, _ = parser.parse_known_args()

    # ── Temp directory for generated files ─────────────────────────────────
    work_dir = os.path.join(tempfile.gettempdir(), "three_people_sim")
    os.makedirs(work_dir, exist_ok=True)

    command_file = os.path.join(work_dir, "character_commands.txt")
    config_file  = os.path.join(work_dir, "sim_config.yaml")

    build_command_file(command_file)
    build_config_file(config_file, command_file)

    # ── Set up the World ────────────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)

    # Load scene (warehouse) or use an empty stage
    if SCENE_USD and not FALLBACK_EMPTY_STAGE:
        print(f"[three_people_sim] Loading scene: {SCENE_USD}")
        omni.usd.get_context().open_stage(SCENE_USD)
        for _ in range(20):
            simulation_app.update()
    else:
        print("[three_people_sim] Using an empty stage (no Nucleus scene).")
        # Add a simple ground plane so navigation works
        world.scene.add_default_ground_plane()

    world.reset()

    # ── Bootstrap IRA programmatically ─────────────────────────────────────
    print("[three_people_sim] Initialising IRA …")
    try:
        import isaacsim.replicator.agent.core as ira_core
        # Load the config – this registers characters & commands
        ira_core.load_config(config_file)
        ira_core.setup_simulation()
        print("[three_people_sim] IRA setup_simulation() complete.")
    except Exception as exc:
        print(f"[three_people_sim] IRA core not available or failed: {exc}")
        print("[three_people_sim] Falling back to omni.anim.people direct API …")
        _fallback_spawn_and_command(command_file)

    # ── Run the simulation ──────────────────────────────────────────────────
    total_frames = args.frames
    print(f"[three_people_sim] Running {total_frames} frames …")

    world.play()
    frame = 0
    while simulation_app.is_running() and frame < total_frames:
        world.step(render=True)
        frame += 1
        if frame % 30 == 0:
            elapsed_s = frame / 30.0
            print(f"  frame {frame:>4}/{total_frames}  ({elapsed_s:.1f} s simulated)")

    world.pause()
    print("[three_people_sim] Simulation finished.")


# ---------------------------------------------------------------------------
# Fallback: spawn characters directly via omni.anim.people when IRA core
# is not accessible (e.g., extension not cached yet on first run).
# ---------------------------------------------------------------------------
def _fallback_spawn_and_command(command_file: str) -> None:
    """
    Use omni.anim.people's PeopleManager / CharacterSpawner directly.
    This mirrors how IRA internally drives the characters.
    """
    try:
        from omni.anim.people.scripts.utils import (
            set_character_commands_file,
        )
        from omni.anim.people.ui_components.command_generation import (
            CharacterCommandGenerator,
        )
        print("[three_people_sim] omni.anim.people fallback: setting command file …")
        set_character_commands_file(command_file)
    except Exception as exc2:
        print(f"[three_people_sim] Fallback also failed: {exc2}")
        print(
            "[three_people_sim] Characters may not animate — "
            "ensure 'omni.anim.people' is enabled and assets are cached."
        )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
        print("[three_people_sim] App closed. Bye!")
