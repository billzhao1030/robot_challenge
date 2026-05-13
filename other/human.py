"""
Setup Three Walkable Human Agents in Isaac Sim 4.5.0
=====================================================
Based on:
  - https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/ext_replicator-agent/agent_control.html
  - https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/ext_replicator-agent/ext_omni_anim_people.html

This script:
  1. Launches Isaac Sim
  2. Enables the required extensions (omni.anim.people + IRA)
  3. Creates a simple ground plane with a NavMesh volume
  4. Builds the NavMesh so characters can walk
  5. Spawns three human characters at different positions
  6. Assigns GoTo / Idle / LookAround commands so they walk around
  7. Attaches Behavior Scripts + Animation Graphs to each character (Setup Characters)
  8. Starts the simulation

Run from Isaac Sim's Python environment:
    ./python.sh setup_three_human_agents.py
Or from the Script Editor inside Isaac Sim (skip the SimulationApp block).
"""

# ---------------------------------------------------------------------------
# 1.  Bootstrap – only needed when running as a standalone script
# ---------------------------------------------------------------------------
import sys

STANDALONE = "omni.isaac.kit" not in sys.modules  # True when run via python.sh

if STANDALONE:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

# ---------------------------------------------------------------------------
# 2.  Imports  (must come AFTER SimulationApp starts)
# ---------------------------------------------------------------------------
import carb
import omni.kit.app
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from pxr import UsdGeom, Gf, UsdPhysics, Sdf
import omni.kit.commands

# ---------------------------------------------------------------------------
# 3.  Enable required extensions
# ---------------------------------------------------------------------------
def _update(n=1):
    """Pump the app event loop n times so async tasks complete."""
    for _ in range(n):
        if STANDALONE:
            simulation_app.update()
        else:
            omni.kit.app.get_app().update()

# Load omni.anim.graph.core FIRST and give it time to register its USD schemas.
# The VariablesService inside it fires an async task that calls HasAPI() on
# AnimationGraphAPI. If other anim extensions load before schemas are registered,
# that task raises "not an applied API schema" — harmless but noisy. Pumping 30
# frames here lets the schema registration + async init fully complete before
# any other anim extension touches the API.
enable_extension("omni.anim.graph.core")
_update(30)

enable_extension("omni.anim.people")             # People simulation / character control
_update(10)
enable_extension("omni.anim.navigation.bundle")  # NavMesh building
_update(10)
enable_extension("isaacsim.replicator.agent.core")
enable_extension("isaacsim.replicator.agent.ui")
_update(20)   # let all async init tasks fully settle

# ---------------------------------------------------------------------------
# 4.  Open a fresh stage and set up a ground plane
# ---------------------------------------------------------------------------
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# Ground plane (10 m × 10 m, centred at origin)
ground_path = "/World/GroundPlane"
ground_prim = UsdGeom.Mesh.Define(stage, ground_path)
ground_prim.CreatePointsAttr([
    Gf.Vec3f(-5, -5, 0),
    Gf.Vec3f( 5, -5, 0),
    Gf.Vec3f( 5,  5, 0),
    Gf.Vec3f(-5,  5, 0),
])
ground_prim.CreateFaceVertexCountsAttr([4])
ground_prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground_prim.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)

# Make it a physics collision surface
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(ground_path))

# ---------------------------------------------------------------------------
# 5.  Add a NavMesh Volume that covers the ground plane
#     Characters use the NavMesh to navigate and avoid obstacles.
#     We define it directly via the USD API to avoid CreatePrim attribute bugs.
# ---------------------------------------------------------------------------
nav_mesh_path = "/World/NavMeshVolume"

# Define a Cube prim via USD API (no omni.kit.commands needed)
nav_cube = UsdGeom.Cube.Define(stage, nav_mesh_path)
nav_prim = stage.GetPrimAtPath(nav_mesh_path)

# Set scale (5 m half-extent → 10 m total on X/Y) and lift centre to Z=1
xform = UsdGeom.Xformable(nav_prim)
xform.ClearXformOpOrder()
translate_op = xform.AddTranslateOp()
translate_op.Set(Gf.Vec3d(0.0, 0.0, 1.0))
scale_op = xform.AddScaleOp()
scale_op.Set(Gf.Vec3f(10.0, 10.0, 2.0))

# Tag the volume so the navigation extension recognises it as a NavMesh volume
nav_prim.SetCustomDataByKey("NavMeshVolume", True)

# Make the volume invisible (it is a logical boundary only)
imageable = UsdGeom.Imageable(nav_prim)
imageable.MakeInvisible()

# ---------------------------------------------------------------------------
# 6.  Build the NavMesh
#     omni.anim.navigation exposes a Python API to bake the mesh.
# ---------------------------------------------------------------------------
try:
    import omni.anim.navigation.core as nav_core
    nav_interface = nav_core.acquire_interface()
    nav_interface.start_navmesh_baking()
    print("[setup_agents] NavMesh baking started.")
except Exception as e:
    print(f"[setup_agents] Could not auto-bake NavMesh via API: {e}")
    print("  → Open the UI: Create > Navigation > NavMeshVolume, then bake manually.")

for _ in range(5):
    _update()

# ---------------------------------------------------------------------------
# 7.  Define the three characters and their command sequences
#
#     Command format for omni.anim.people:
#       Spawn <name> <x> <y> <z> <rotation_deg>
#       <name> GoTo  <x> <y> <z> <rotation_deg|_>
#       <name> Idle  <duration_sec>
#       <name> LookAround <duration_sec>
#
#     Character asset names must match files under:
#       omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/People/Characters/
#     Using generic names (no exact match) causes a random asset to be loaded.
# ---------------------------------------------------------------------------
COMMANDS = """\
# Spawn three human agents at different starting positions
Spawn Character_01  0.0  0.0 0  0
Spawn Character_02  2.0  0.0 0  90
Spawn Character_03 -2.0  0.0 0  180

# Character_01: walk in a triangle, pause, look around, repeat
Character_01 GoTo  3.0  3.0 0 _
Character_01 Idle 3
Character_01 GoTo -3.0  3.0 0 _
Character_01 LookAround 4
Character_01 GoTo  0.0 -3.0 0 _
Character_01 Idle 2

# Character_02: walk back and forth on X axis
Character_02 GoTo  4.0  0.0 0 0
Character_02 Idle 2
Character_02 GoTo -4.0  0.0 0 180
Character_02 Idle 2
Character_02 GoTo  0.0  0.0 0 0

# Character_03: wander in a square
Character_03 GoTo  2.0  2.0 0 _
Character_03 LookAround 3
Character_03 GoTo  2.0 -2.0 0 _
Character_03 Idle 2
Character_03 GoTo -2.0 -2.0 0 _
Character_03 LookAround 3
Character_03 GoTo -2.0  2.0 0 _
Character_03 Idle 2
"""

# ---------------------------------------------------------------------------
# 8.  Feed commands to the omni.anim.people extension
# ---------------------------------------------------------------------------
try:
    from omni.anim.people.ui_components.command_setting_panel import CommandSettingPanel  # noqa

    # Try the public Python API first (Isaac Sim 4.x)
    import omni.anim.people as anim_people
    people_interface = anim_people.get_interface()   # may raise if not available

    # Write commands to a temp file and point the extension at it
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write(COMMANDS)
    tmp.close()

    people_interface.set_command_file_path(tmp.name)
    people_interface.load_characters()   # loads USD assets + animation graphs
    people_interface.setup_characters()  # attaches BehaviorScript + AnimGraph

    print(f"[setup_agents] Commands loaded from: {tmp.name}")

except Exception as e:
    # Fallback: use carb settings to set the command text directly
    print(f"[setup_agents] Direct API unavailable ({e}). Using carb settings fallback.")
    settings = carb.settings.get_settings()
    settings.set("/exts/omni/anim/people/command_text", COMMANDS)

    # Attempt to trigger load/setup via the UI backend
    try:
        import omni.anim.people.scripts.utils as people_utils
        people_utils.load_characters_from_command_text(COMMANDS)
        people_utils.setup_characters()
    except Exception as e2:
        print(f"[setup_agents] Fallback also failed: {e2}")
        print("  → Use the People Simulation UI (Window > People Simulation):")
        print("    1. Paste the commands below into the Command Text Box.")
        print("    2. Click 'Load Characters'.")
        print("    3. Click 'Setup Characters'.")
        print("    4. Ensure 'Navmesh Based Navigation' is ON.")
        print("    5. Press Play.")
        print()
        print(COMMANDS)

# ---------------------------------------------------------------------------
# 9.  Enable NavMesh-based navigation (so agents walk on the ground)
# ---------------------------------------------------------------------------
settings = carb.settings.get_settings()
settings.set("/exts/omni/anim/people/navmesh_enabled", True)

# ---------------------------------------------------------------------------
# 10. Run the simulation
# ---------------------------------------------------------------------------
print("[setup_agents] Starting simulation. Press Escape / stop to end.")

if STANDALONE:
    from omni.isaac.core import SimulationContext
    sim_context = SimulationContext()
    sim_context.play()

    frame = 0
    try:
        while simulation_app.is_running():
            simulation_app.update()
            frame += 1
            if frame % 300 == 0:
                elapsed = frame / 60  # approximate seconds at 60 fps
                print(f"[setup_agents] Simulation running… {elapsed:.0f}s")
    except KeyboardInterrupt:
        pass
    finally:
        sim_context.stop()
        simulation_app.close()
else:
    # Running inside Isaac Sim Script Editor – just press Play in the toolbar
    print("[setup_agents] Script loaded in Script Editor. Press Play in Isaac Sim to start.")