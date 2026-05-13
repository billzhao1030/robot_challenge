from isaacsim import SimulationApp

# Start Isaac Sim
simulation_app = SimulationApp({"headless": False})

import carb
import omni

# Extension enabling helper differs across versions; this pattern appears in Isaac Sim forum examples :contentReference[oaicite:14]{index=14}
try:
    from isaacsim.core.utils.extensions import enable_extension
except Exception:
    from omni.isaac.core.utils.extensions import enable_extension

# Enable IRA (core is typically enough for scripting; UI optional)
enable_extension("isaacsim.replicator.agent.core")

# You usually also need these for character animation backend
enable_extension("omni.anim.graph.core")
enable_extension("omni.anim.graph.schema")
enable_extension("omni.anim.navigation")

# ---- Load a scene USD (must have NavMesh baked or bake-able) ----
from omni.isaac.core.utils.stage import open_stage

ASSET_ROOT = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
stage_path = f"{ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
open_stage(stage_path)

# ---- Spawn and setup characters ----
from isaacsim.replicator.agent.core.simulation import SimulationManager  # per NVIDIA forum :contentReference[oaicite:15]{index=15}

sim_mgr = SimulationManager()

# Spawn 3 people at fixed spots (idx selects which character asset preset)
sim_mgr.spawn_character_by_idx(spawn_location=(0.0, 0.0, 0.0), spawn_rotation=(0.0, 0.0, 0.0), idx=0)
sim_mgr.spawn_character_by_idx(spawn_location=(2.0, 0.0, 0.0), spawn_rotation=(0.0, 0.0, 90.0), idx=1)
sim_mgr.spawn_character_by_idx(spawn_location=(0.0, 2.0, 0.0), spawn_rotation=(0.0, 0.0, -90.0), idx=2)

sim_mgr.setup_all_characters()

# ---- Now drive behavior via IRA command file approach (recommended) ----
# In practice, the simplest/most robust way is still: generate a command file and let IRA execute it.
# If you want real-time overrides, IRA supports "command injection" conceptually from UI :contentReference[oaicite:16]{index=16}
# (programmatic injection API is not clearly documented in 4.5 docs).

# Basic simulation loop
timeline = omni.timeline.get_timeline_interface()
timeline.play()

while simulation_app.is_running():
    simulation_app.update()

timeline.stop()
simulation_app.close()