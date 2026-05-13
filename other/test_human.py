# # test_03_spawn_people.py
# from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": False})

# import omni.usd
# import omni.kit.commands
# from omni.isaac.core import World
# from omni.isaac.core.utils.nucleus import get_assets_root_path
# from pxr import UsdGeom, Gf

# assets_root = get_assets_root_path()

# # 3 character USDs using corrected paths
# CHARACTERS = [
#     {
#         "name": "Mom",
#         "usd": f"{assets_root}/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
#         "prim_path": "/World/People/Mom",
#         "position": (2.0, 0.0, 0.0),
#     },
#     {
#         "name": "Dad",
#         "usd": f"{assets_root}/Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
#         "prim_path": "/World/People/Dad",
#         "position": (0.0, 2.0, 0.0),
#     },
#     {
#         "name": "Child",
#         "usd": f"{assets_root}/Isaac/People/Characters/biped_demo/biped_demo_meters.usd",
#         "prim_path": "/World/People/Child",
#         "position": (-2.0, 0.0, 0.0),
#     },
# ]

# def spawn_person(usd_path, prim_path, position):
#     omni.kit.commands.execute(
#         "CreateReferenceCommand",
#         usd_context=omni.usd.get_context(),
#         path_to=prim_path,
#         asset_path=usd_path,
#         instanceable=False,
#     )
#     stage = omni.usd.get_context().get_stage()
#     prim = stage.GetPrimAtPath(prim_path)
#     if not prim.IsValid():
#         print(f"  ❌ Prim not found at {prim_path}")
#         return False
#     xform = UsdGeom.Xformable(prim)
#     xform.ClearXformOpOrder()
#     xform.AddTranslateOp().Set(Gf.Vec3d(*position))
#     print(f"  ✅ Spawned at {position}")
#     return True

# # Setup world
# world = World()
# world.scene.add_default_ground_plane()

# # Spawn all 3 people
# print("Spawning characters...")
# for char in CHARACTERS:
#     print(f"  Loading {char['name']}...")
#     spawn_person(char["usd"], char["prim_path"], char["position"])

# world.reset()
# print("Running sim — check the viewport for 3 people!")

# step = 0
# while simulation_app.is_running():
#     world.step(render=True)
#     step += 1
#     if step % 500 == 0:
#         print(f"  Step {step} running...")

# simulation_app.close()

# test_04_animated_people.py
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni
import omni.usd
import omni.kit.commands
from pxr import Sdf, UsdGeom, Gf, UsdSkel
from omni.isaac.core.world import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.utils.nucleus import get_assets_root_path

# ── 1. Enable extensions ─────────────────────────────────────
print("[INFO] Enabling extensions...")
extensions.enable_extension("omni.anim.people")
extensions.enable_extension("omni.anim.graph.bundle")
extensions.enable_extension("omni.kit.scripting")
for _ in range(30):
    simulation_app.update()
print("[INFO] Extensions ready")

# ── 2. Paths ─────────────────────────────────────────────────
assets_root = get_assets_root_path()

BEHAVIOR_SCRIPT = "/home/xunyi/isaacsim4.5/extscache/omni.anim.people-0.6.7+106.5.0/omni/anim/people/scripts/character_behavior.py"
BIPED_USD       = f"{assets_root}/Isaac/People/Characters/Biped_Setup.usd"

CHARACTERS = [
    {
        "name":      "Mom",
        "usd":       f"{assets_root}/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
        "prim_path": "/World/Characters/Mom",
        "spawn_pos": (2.0, 0.0, 0.0),
        "commands":  [
            "GoTo 5.0 2.0 0.0 _",
            "Idle 3",
            "GoTo 0.0 3.0 0.0 _",
            "Idle 3",
            "GoTo 2.0 0.0 0.0 _",
        ]
    },
    {
        "name":      "Dad",
        "usd":       f"{assets_root}/Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
        "prim_path": "/World/Characters/Dad",
        "spawn_pos": (0.0, 0.0, 0.0),
        "commands":  [
            "GoTo -3.0 2.0 0.0 _",
            "Idle 4",
            "GoTo 2.0 -2.0 0.0 _",
            "Idle 2",
        ]
    },
    {
        "name":      "Child",
        "usd":       f"{assets_root}/Isaac/People/Characters/biped_demo/biped_demo_meters.usd",
        "prim_path": "/World/Characters/Child",
        "spawn_pos": (-2.0, 0.0, 0.0),
        "commands":  [
            "GoTo 1.0 4.0 0.0 _",
            "Idle 2",
            "GoTo -4.0 -1.0 0.0 _",
            "Idle 2",
        ]
    },
]

# ── 3. World + ground ────────────────────────────────────────
world = World()
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
prims.create_prim("/World/Characters", "Xform")

# ── 4. Biped_Setup (provides the animation graph) ────────────
print("[INFO] Loading Biped_Setup...")
biped_prim = prims.create_prim(
    "/World/Characters/Biped_Setup", "Xform", usd_path=BIPED_USD
)
biped_prim.GetAttribute("visibility").Set("invisible")
for _ in range(20):
    simulation_app.update()

ANIM_GRAPH_PATH = Sdf.Path(
    "/World/Characters/Biped_Setup/CharacterAnimation/AnimationGraph"
)
anim_graph_prim = stage.GetPrimAtPath(ANIM_GRAPH_PATH)
if anim_graph_prim.IsValid():
    print(f"✅ AnimationGraph found at {ANIM_GRAPH_PATH}")
else:
    # Walk the Biped_Setup tree to find it
    print("AnimationGraph not at expected path, searching...")
    def find_prim_by_type(root_path, type_name):
        prim = stage.GetPrimAtPath(root_path)
        if type_name.lower() in prim.GetTypeName().lower():
            return prim.GetPath()
        for child in prim.GetAllChildren():
            result = find_prim_by_type(child.GetPath(), type_name)
            if result:
                return result
        return None
    found = find_prim_by_type("/World/Characters/Biped_Setup", "AnimationGraph")
    if found:
        ANIM_GRAPH_PATH = Sdf.Path(found)
        print(f"✅ AnimationGraph found at {ANIM_GRAPH_PATH}")
    else:
        print("❌ AnimationGraph not found — listing Biped_Setup tree:")
        def print_tree(path, indent=0):
            prim = stage.GetPrimAtPath(path)
            print(f"{'  '*indent}{prim.GetName()} ({prim.GetTypeName()})")
            for child in prim.GetAllChildren():
                print_tree(child.GetPath(), indent+1)
        print_tree("/World/Characters/Biped_Setup")

# ── 5. Helper: find SkelRoot ─────────────────────────────────
def find_skelroot(base_path):
    prim = stage.GetPrimAtPath(base_path)
    if not prim.IsValid():
        return None
    if prim.IsA(UsdSkel.Root):
        return prim.GetPath()
    for child in prim.GetAllChildren():
        result = find_skelroot(child.GetPath())
        if result:
            return result
    return None

# ── 6. Spawn characters ──────────────────────────────────────
for char in CHARACTERS:
    print(f"\n[INFO] Spawning {char['name']}...")

    char_prim = prims.create_prim(
        char["prim_path"], "Xform", usd_path=char["usd"]
    )
    xform = UsdGeom.Xformable(char_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*char["spawn_pos"]))

    for _ in range(15):
        simulation_app.update()

    skelroot_path = find_skelroot(char["prim_path"])
    if not skelroot_path:
        print(f"  ❌ SkelRoot not found under {char['prim_path']}")
        char["skelroot_path"] = None
        continue
    print(f"  ✅ SkelRoot: {skelroot_path}")
    char["skelroot_path"] = str(skelroot_path)

    # Attach behavior script
    omni.kit.commands.execute(
        "ApplyScriptingAPICommand",
        paths=[skelroot_path]
    )
    skelroot_prim = stage.GetPrimAtPath(skelroot_path)
    skelroot_prim.GetAttribute("omni:scripting:scripts").Set(
        Sdf.AssetPathArray([BEHAVIOR_SCRIPT])
    )

    # Attach animation graph
    omni.kit.commands.execute(
        "ApplyAnimationGraphAPICommand",
        paths=[skelroot_path],
        animation_graph_path=ANIM_GRAPH_PATH
    )
    print(f"  ✅ Animation graph attached")

# ── 7. Reset and inject walk commands ───────────────────────
print("\n[INFO] Resetting world...")
world.reset()
for _ in range(60):
    simulation_app.update()

try:
    from omni.anim.people.scripts.global_agent_manager import GlobalAgentManager
    from omni.anim.people.scripts.character_behavior import CharacterBehavior

    agent_manager = GlobalAgentManager()

    for char in CHARACTERS:
        if not char.get("skelroot_path"):
            continue

        skel_path = char["skelroot_path"]
        agent = CharacterBehavior(prim_path=Sdf.Path(skel_path))
        agent_manager.add_agent(agent_prim_path=skel_path, agent_object=agent)

        # Commands format: "CharacterName Action params"
        cmd_list = [f"{char['name']} {cmd}" for cmd in char["commands"]]
        agent_manager.inject_command(
            agent_prim_path=skel_path,
            command_list=cmd_list
        )
        print(f"[INFO] ✅ Commands injected for {char['name']}: {cmd_list}")

except Exception as e:
    print(f"[WARN] Command injection failed: {e}")
    import traceback
    traceback.print_exc()

# ── 8. Run ───────────────────────────────────────────────────
print("\n[INFO] Running — watch the viewport!")
step = 0
while simulation_app.is_running():
    world.step(render=True)
    step += 1
    if step % 500 == 0:
        print(f"  Step {step}")

simulation_app.close()