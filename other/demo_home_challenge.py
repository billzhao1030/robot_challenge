"""
demo_home_challenge.py
======================
室内一日挑战 Demo — Isaac Sim 4.5

场景:
  - 加载 InteriorAgent 室内 USD 场景
  - 生成 Unitree G1 机器人
  - 生成家庭三口（爸爸/妈妈/小孩）作为 ghost-like Capsule 人物
  - 模拟一天时间（时间加速）
  - 家庭成员定时生成请求，机器人大脑自主决策 & 执行导航

用法:
  python demo_home_challenge.py [--headless] [--family nuclear|large|elderly|full]
                                [--time_scale 720] [--scene kujiale_0003]

控制:
  摄像头 W/S/A/D 移动（来自 demo.py 的 FloatingCameraController）
  R 键重置场景
"""

import argparse
import json
import os
import sys
import numpy as np

# ── 命令行参数（必须在 SimulationApp 之前）──────────────────────────────── #
parser = argparse.ArgumentParser(description="室内一日挑战 Isaac Sim Demo")
parser.add_argument("--headless",    default=False, action="store_true")
parser.add_argument("--family",      default="nuclear",
                    choices=["nuclear", "large", "elderly", "full"])
parser.add_argument("--time_scale",  default=720.0, type=float,
                    help="时间加速倍率（默认720：约2分钟真实时间=1天虚拟时间）")
parser.add_argument("--scene",       default="kujiale_0003",
                    help="数据集场景 ID，例如 kujiale_0003")
parser.add_argument("--camera_height", default=1.5, type=float)
args, _unknown = parser.parse_known_args()

# ── 启动 Isaac Sim ─────────────────────────────────────────────────────── #
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "renderer": "RayTracedLighting",
    "headless": args.headless,
})

import carb
from pxr import UsdGeom, Sdf, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage

# 自制模块
sys.path.insert(0, os.path.dirname(__file__))
from time_manager   import TimeManager
from family_simulator import FamilySimulator, load_room_centers
from robot_brain    import RobotBrain, RobotState

# ── 路径配置 ─────────────────────────────────────────────────────────────── #
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
SCENE_DIR  = os.path.join(DATA_DIR, args.scene)
USD_PATH   = os.path.join(SCENE_DIR, f"{args.scene}.usda")
ROOMS_JSON = os.path.join(SCENE_DIR, "rooms.json")

# Unitree G1 robot USD（NVIDIA Isaac Sim 内置路径）
# 如果本地没有，使用 Nucleus 拉取或 Omniverse 内置
G1_USD_CANDIDATES = [
    "/home/xunyi/.local/share/ov/pkg/isaacsim-4.5.0/exts/isaacsim.robot.policy.examples/data/robots/g1_23dof_lock_waist.usd",
    "/home/xunyi/.local/share/ov/pkg/isaacsim-4.5.0/exts/isaacsim.robot.policy.examples/data/robots/g1.usd",
    "omniverse://localhost/NVIDIA/Assets/IsaacLab/Robots/Unitree/G1/g1_minimal.usd",
    "omniverse://localhost/NVIDIA/Assets/Robots/Unitree/G1/g1.usd",
]

def find_g1_usd() -> str | None:
    """尝试找到本地或 Nucleus 上的 G1 USD"""
    for p in G1_USD_CANDIDATES:
        if p.startswith("omniverse://"):
            # 无法直接检测 Nucleus 文件，先跳过
            continue
        if os.path.exists(p):
            print(f"[G1] 找到本地 G1 USD: {p}")
            return p
    # 搜索本地可能路径
    search_roots = [
        "/home/xunyi/.local/share/ov",
        "/home/xunyi/isaacsim4.5/exts",
        "/home/xunyi/isaacsim4.5/extscache",
    ]
    import glob
    for root in search_roots:
        hits = glob.glob(f"{root}/**/g1*.usd", recursive=True)[:3]
        hits += glob.glob(f"{root}/**/G1*.usd", recursive=True)[:3]
        for h in hits:
            print(f"[G1] 找到本地 G1 USD: {h}")
            return h
    print("[G1] 警告: 未找到 G1 USD，将使用占位方块代替")
    return None


# ═══════════════════════════════════════════════════════════════════════════ #
#  Ghost 人物创建工具                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def create_ghost_person(stage, prim_path: str, position: tuple,
                        color: tuple = (0.5, 0.5, 0.9),
                        height: float = 1.7,
                        name: str = "Person"):
    """
    在场景中创建一个简单的胶囊体代表家庭成员 (ghost-like)。
    包含:
      - 躯干 Capsule
      - 头部 Sphere
      - 名称标签 (通过 print 说明，USD 无内置文字)
    """
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf

    # 根 Xform
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform_prim = xform.GetPrim()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    # 躯干 Capsule
    body_path = prim_path + "/Body"
    body = UsdGeom.Capsule.Define(stage, body_path)
    body.CreateRadiusAttr(0.18)
    body.CreateHeightAttr(height * 0.55)
    body.CreateAxisAttr("Z")
    body_xform = UsdGeom.Xformable(body.GetPrim())
    body_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, height * 0.45))

    # 头部 Sphere
    head_path = prim_path + "/Head"
    head = UsdGeom.Sphere.Define(stage, head_path)
    head.CreateRadiusAttr(0.125)
    head_xform = UsdGeom.Xformable(head.GetPrim())
    head_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, height * 0.85))

    # 染色（通过 DisplayColor）
    display_color = Gf.Vec3f(*color)
    for geom in [body, head]:
        geom.CreateDisplayColorAttr([display_color])

    # 半透明材质（ghost 效果）
    _apply_ghost_material(stage, prim_path, color)

    print(f"  [Scene] 创建人物 {name} @ {position}")
    return xform_prim


def _apply_ghost_material(stage, prim_path: str, color: tuple):
    """给人物 prim 树挂上半透明材质"""
    from pxr import UsdShade, Gf
    mat_path = prim_path + "/GhostMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # 绑定到根 xform 下所有子 prim
    from pxr import UsdGeom
    root_prim = stage.GetPrimAtPath(prim_path)
    if root_prim:
        binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
        binding_api.Bind(mat)
        for child in root_prim.GetChildren():
            b = UsdShade.MaterialBindingAPI.Apply(child)
            b.Bind(mat)


def move_ghost_person(stage, prim_path: str, position: tuple):
    """更新 ghost 人物位置"""
    from pxr import Gf
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position))


# ═══════════════════════════════════════════════════════════════════════════ #
#  G1 机器人 Prim 创建                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

def create_g1_robot(stage, world, g1_usd: str | None,
                    position: tuple = (0.0, 0.0, 0.1)) -> str:
    """
    加载 G1 机器人到场景。
    若找不到 USD，用一个带颜色的简单形状代替（占位）。
    返回 prim_path。
    """
    robot_path = "/World/G1Robot"

    if g1_usd and os.path.exists(g1_usd):
        prim = define_prim(robot_path, "Xform")
        prim.GetReferences().AddReference(g1_usd)
        # 位移
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position))
        print(f"[G1] 机器人 USD 已加载: {g1_usd}")
    else:
        # 占位: 蓝色胶囊体
        _create_robot_placeholder(stage, robot_path, position)

    return robot_path


def _create_robot_placeholder(stage, prim_path: str, position: tuple):
    """G1 未找到时，用简单几何体占位"""
    from pxr import Gf, UsdShade

    xform = UsdGeom.Xform.Define(stage, prim_path)
    UsdGeom.Xformable(xform).AddTranslateOp().Set(Gf.Vec3d(*position))

    # 身体
    body = UsdGeom.Capsule.Define(stage, prim_path + "/Body")
    body.CreateRadiusAttr(0.2)
    body.CreateHeightAttr(0.9)
    body.CreateAxisAttr("Z")
    body_xform = UsdGeom.Xformable(body.GetPrim())
    body_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.85))
    body.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.4, 0.9)])

    # 头
    head = UsdGeom.Sphere.Define(stage, prim_path + "/Head")
    head.CreateRadiusAttr(0.14)
    UsdGeom.Xformable(head.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.6))
    head.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.55, 0.95)])

    # 眼睛（LED 感）
    for side, y in [("LEye", 0.06), ("REye", -0.06)]:
        eye = UsdGeom.Sphere.Define(stage, f"{prim_path}/{side}")
        eye.CreateRadiusAttr(0.025)
        UsdGeom.Xformable(eye.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.13, y, 1.63))
        eye.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.9)])

    print(f"[G1] 使用占位机器人（G1 USD 未找到）@ {position}")


def move_robot(stage, robot_path: str, position: tuple):
    """平滑移动机器人（直接设置 translate，实际需接入物理控制器）"""
    from pxr import Gf
    prim = stage.GetPrimAtPath(robot_path)
    if prim and prim.IsValid():
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position))


# ═══════════════════════════════════════════════════════════════════════════ #
#  HUD 数据（打印到终端，可扩展为 omni.ui 面板）                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def print_hud(time_mgr: TimeManager, brain: RobotBrain, family: FamilySimulator, frame: int):
    if frame % 240 != 0:   # 每240帧打印一次（约2秒）
        return
    t = time_mgr.summary()
    b = brain.status_summary()
    mpos = {m.member_id: m.current_room for m in family.members}
    print(
        f"\n{'='*60}\n"
        f"  ⏰ 虚拟时间: {t['time']} ({t['period_zh']})  进度: {t['progress']*100:.1f}%\n"
        f"  🤖 机器人: {b['state']}  电量: {b['battery']}%  "
        f"已完成: {b['tasks_done']}  队列: {b['queue_len']}\n"
        f"     当前任务: {b['current_task'] or '—'}\n"
        f"  👨‍👩‍👦 家庭:\n"
        + "\n".join(f"     {m.name_zh}: {mpos[m.member_id]}" for m in family.members)
        + f"\n{'='*60}"
    )


# ═══════════════════════════════════════════════════════════════════════════ #
#  摄像机控制（复用 demo.py 的逻辑）                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class FloatingCameraController:
    def __init__(self, stage, start_pos: tuple, camera_height: float = 1.5):
        self.stage = stage
        self.camera_path = "/World/FloatingCamera"
        self.camera_height = camera_height
        self.position = np.array([start_pos[0], start_pos[1], camera_height])
        self.look_dir = np.array([1.0, 0.0, 0.0])
        self.move_speed = 1.5
        self.turn_speed = 40.0
        self._cmd = np.zeros(3)
        self._init_prim()

    def _init_prim(self):
        cam = UsdGeom.Camera.Define(self.stage, self.camera_path)
        cam.CreateFocalLengthAttr(10.0)
        cam.CreateClippingRangeAttr().Set((0.01, 1000.0))
        cam.CreateVerticalApertureAttr(20.0)
        cam.CreateHorizontalApertureAttr(20.0)
        self._update()

    def subscribe_keyboard(self):
        import omni.appwindow
        self._aw  = omni.appwindow.get_default_app_window()
        self._inp = carb.input.acquire_input_interface()
        self._kb  = self._aw.get_keyboard()
        self._inp.subscribe_to_keyboard_events(self._kb, self._on_key)

    def _on_key(self, event, *a):
        from carb.input import KeyboardEventType
        mapping = {
            "W": np.array([1, 0, 0]),
            "S": np.array([-1, 0, 0]),
            "A": np.array([0, 0, 1]),
            "D": np.array([0, 0, -1]),
        }
        if event.input.name in mapping:
            d = mapping[event.input.name]
            if event.type == KeyboardEventType.KEY_PRESS:
                self._cmd += d
            elif event.type == KeyboardEventType.KEY_RELEASE:
                self._cmd -= d
        return True

    def step(self, dt: float):
        from scipy.spatial.transform import Rotation as R
        if self._cmd[0]:
            self.position += self.look_dir * self._cmd[0] * self.move_speed * dt
        if self._cmd[2]:
            angle = self._cmd[2] * self.turn_speed * dt
            self.look_dir = R.from_euler("z", angle, degrees=True).apply(self.look_dir)
        self.position[2] = self.camera_height
        self._update()

    def _update(self):
        from scipy.spatial.transform import Rotation as R
        prim = self.stage.GetPrimAtPath(self.camera_path)
        if not (prim and prim.IsValid()):
            return
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.position))
        fwd = self.look_dir / (np.linalg.norm(self.look_dir) + 1e-9)
        up  = np.array([0, 0, 1])
        right = np.cross(up, fwd)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1, 0, 0])
        up2  = np.cross(fwd, right)
        mat  = np.column_stack([fwd, right, up2])
        from scipy.spatial.transform import Rotation as R
        q = R.from_matrix(mat).as_euler("xyz", degrees=True)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(q[0] + 90, q[1], q[2] - 90))


# ═══════════════════════════════════════════════════════════════════════════ #
#  主运行函数                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def run():
    print("\n" + "="*60)
    print("  🏠 室内一日挑战 Demo — Isaac Sim 4.5")
    print(f"  场景: {args.scene}   家庭类型: {args.family}")
    print(f"  时间加速: {args.time_scale}x")
    print("="*60 + "\n")

    # ── 创建 World ─────────────────────────────────────────────────────── #
    world = World(stage_units_in_meters=1.0,
                  physics_dt=1.0 / 120.0,
                  rendering_dt=8.0 / 200.0)
    world.scene.add_default_ground_plane(z_position=0.0,
                                          name="ground",
                                          prim_path="/World/defaultGroundPlane")

    stage = simulation_app.context.get_stage()

    # ── 加载室内场景 ───────────────────────────────────────────────────── #
    if os.path.exists(USD_PATH):
        prim = define_prim("/World/IndoorScene", "Xform")
        prim.GetReferences().AddReference(USD_PATH, "/Root")
        print(f"[Scene] 室内场景已加载: {USD_PATH}")
    else:
        print(f"[Scene] 警告: 未找到场景 USD: {USD_PATH}")

    # ── 环境光照 ────────────────────────────────────────────────────────── #
    dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(500.0)

    # HDR 环境贴图
    hdr_path = os.path.join(SCENE_DIR, "buikslotermeerplein_4k.hdr")
    if os.path.exists(hdr_path):
        dome.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(hdr_path)

    # ── 加载房间信息 ───────────────────────────────────────────────────── #
    room_centers = {}
    if os.path.exists(ROOMS_JSON):
        room_centers = load_room_centers(ROOMS_JSON)
        print(f"[Scene] 房间: {list(room_centers.keys())}")
    else:
        print("[Scene] 警告: rooms.json 未找到，使用默认坐标")
        room_centers = {
            "living_room": (0.0,  0.0, 0.1),
            "kitchen":     (-4.0, 4.0, 0.1),
            "bedroom":     (5.0,  -2.0, 0.1),
        }

    # 找一个合适的起始位置（living_room 附近）
    start_pos = room_centers.get("living_room", (0.0, 0.0, 0.1))

    # ── G1 机器人 ──────────────────────────────────────────────────────── #
    g1_usd = find_g1_usd()
    robot_offset = (start_pos[0], start_pos[1] + 1.0, 0.1)
    robot_path = create_g1_robot(stage, world, g1_usd, position=robot_offset)

    # ── 家庭成员 (Ghost People) ────────────────────────────────────────── #
    family = FamilySimulator(ROOMS_JSON if os.path.exists(ROOMS_JSON) else None
                              or ROOMS_JSON, family_type=args.family)

    ghost_paths = {}
    for member in family.members:
        gpath = f"/World/Family/{member.member_id}"
        pos = member.position
        create_ghost_person(stage, gpath, pos,
                            color=member.color,
                            height=member.height,
                            name=member.name_zh)
        ghost_paths[member.member_id] = gpath

    # ── 摄像机 ─────────────────────────────────────────────────────────── #
    cam_ctrl = FloatingCameraController(stage, start_pos, args.camera_height)
    if not args.headless:
        try:
            cam_ctrl.subscribe_keyboard()
        except Exception as e:
            print(f"[Cam] 键盘订阅失败 (headless?): {e}")

    # ── 时间管理 & 机器人大脑 ──────────────────────────────────────────── #
    time_mgr = TimeManager(sim_dt=1.0/120.0,
                            time_scale=args.time_scale,
                            start_hour=6.0)    # 从早上6:00开始
    brain    = RobotBrain(room_centers)

    # 机器人当前位置（仿真层面）
    robot_sim_pos = np.array(robot_offset, dtype=float)

    # ── 物理 Callback ─────────────────────────────────────────────────── #
    first_step = [True]

    def on_physics_step(step_dt: float):
        nonlocal robot_sim_pos
        if first_step[0]:
            first_step[0] = False
            return

        # 1. 推进虚拟时间
        time_mgr.step(step_dt)

        # 2. 家庭模拟
        family.step(time_mgr)

        # 3. 更新 ghost 人物 USD 位置
        for member in family.members:
            gp = ghost_paths.get(member.member_id)
            if gp:
                move_ghost_person(stage, gp, member.position)

        # 4. 批量取请求，喂给机器人大脑
        new_reqs = family.pop_requests()
        action = brain.update(time_mgr, new_reqs)

        # 5. 模拟机器人导航（简单线性插值，实际需要寻路控制器）
        if action["action"] in ("navigate", "execute"):
            tgt = np.array(action["target_pos"])
            delta = tgt - robot_sim_pos
            dist = np.linalg.norm(delta)
            if dist > 0.05:
                # 每步最多移动 0.3m（结合时间加速）
                step_move = min(0.3 * args.time_scale / 120.0 * step_dt * 120.0, dist)
                robot_sim_pos = robot_sim_pos + delta / dist * step_move
                move_robot(stage, robot_path, tuple(robot_sim_pos))
            else:
                brain.position = robot_sim_pos.copy()

        # 6. 摄像机
        cam_ctrl.step(step_dt)

    world.add_physics_callback("home_challenge_step", on_physics_step)

    # ── 主循环 ────────────────────────────────────────────────────────── #
    world.reset()
    frame = 0

    print("\n[Sim] 开始仿真！摄像机控制: W/S/A/D | 关闭窗口退出")
    print(f"[Sim] 今日家庭: {[m.name_zh for m in family.members]}\n")

    while simulation_app.is_running():
        world.step(render=True)

        if world.is_playing():
            print_hud(time_mgr, brain, family, frame)
            frame += 1

            # 一天结束
            if time_mgr.progress() >= 1.0:
                print("\n[Sim] 虚拟一天结束！统计摘要:")
                print(f"  已完成任务: {brain.tasks_completed}")
                print(f"  剩余队列: {len(brain.task_queue)}")
                print(f"  最终电量: {brain.battery:.1f}%")
                break

    print("\n[Sim] 仿真结束，感谢使用室内一日挑战 Demo！")


# ═══════════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    run()
    simulation_app.close()
