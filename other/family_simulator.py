"""
FamilySimulator: 模拟家庭成员的行为与请求

家庭成员类型:
  - 爸爸 (dad)
  - 妈妈 (mom)
  - 孩子 (child)  -- 可以有多个
  - 老人 (elder)  -- 可选

每个成员:
  - 有当前位置（来自 rooms.json 的房间中心）
  - 在特定时间段会产生请求（自然语言）
  - 请求通过 GPT 或规则模板生成

请求示例:
  "把厨房里的杯子拿给我"
  "帮我把脏衣服放到洗衣机里"
  "我有点渴，能给我倒杯水吗"
"""

import json
import math
import random
import os
from typing import Optional
from time_manager import TimeManager


# ---------------------------------------------------------------------------
# 房间中心坐标工具
# ---------------------------------------------------------------------------

def polygon_centroid(polygon: list) -> tuple:
    """计算多边形质心 (x, y)"""
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return (cx, cy)


def load_room_centers(rooms_json_path: str) -> dict:
    """从 rooms.json 载入各房间中心坐标"""
    with open(rooms_json_path, "r") as f:
        rooms = json.load(f)

    centers = {}
    # 同类型房间用编号区分
    type_count = {}
    for room in rooms:
        rt = room["room_type"].lower().replace(" ", "_")
        type_count[rt] = type_count.get(rt, 0) + 1
        key = rt if type_count[rt] == 1 else f"{rt}_{type_count[rt]}"
        cx, cy = polygon_centroid(room["polygon"])
        centers[key] = (cx, cy, 0.1)   # z=0.1 (站在地面上)
    return centers


# ---------------------------------------------------------------------------
# 规则型请求模板
# ---------------------------------------------------------------------------

REQUEST_TEMPLATES = {
    "morning": [
        ("fetch", "厨房", "把厨房里的杯子拿给我"),
        ("fetch", "厨房", "帮我把早餐端过来"),
        ("clean", "living_room", "客厅地板有点脏，帮忙擦一下"),
        ("fetch", "厨房", "我需要一杯热水，谢谢"),
    ],
    "noon": [
        ("fetch", "厨房", "能帮我把厨房的碗筷拿来吗"),
        ("clean", "living_room", "吃完饭了，帮忙收拾一下桌子"),
        ("fetch", "bedroom", "我有点累，帮我把卧室的枕头拿来"),
        ("laundry", "bedroom", "卧室有些脏衣服，帮忙拿去洗"),
    ],
    "afternoon": [
        ("fetch", "living_room", "帮我把遥控器拿来"),
        ("clean", "bathroom", "浴室需要清洁，帮忙打扫一下"),
        ("fetch", "厨房", "我有点饿，能去厨房帮我拿点零食吗"),
        ("water_plants", "balcony", "阳台的花需要浇水了"),
    ],
    "evening": [
        ("fetch", "厨房", "帮我倒一杯水"),
        ("laundry", "bedroom", "帮忙把洗好的衣服晾起来"),
        ("clean", "living_room", "今天客厅比较乱，帮忙收拾一下"),
        ("fetch", "bedroom", "帮我把被子从卧室拿出来"),
    ],
    "night": [
        ("fetch", "bedroom", "帮我把药从床头柜上拿来"),
        ("check", "living_room", "检查一下门窗是否关好"),
        ("fetch", "厨房", "能给我倒杯温水吗"),
    ],
}

# 老人特定请求（较多健康/安全相关）
ELDER_TEMPLATES = {
    "morning": [
        ("fetch", "厨房", "帮我把药拿来，我需要吃早上的药"),
        ("fetch", "厨房", "能帮我拿一杯温水吗，我要吃药"),
        ("check", "living_room", "帮我看看今天的天气怎么样"),
    ],
    "afternoon": [
        ("fetch", "bedroom", "我想休息一下，帮我把毯子拿来"),
        ("clean", "living_room", "客厅地板需要扫一下"),
        ("fetch", "厨房", "能给我一杯热茶吗"),
    ],
    "evening": [
        ("fetch", "bedroom", "帮我把睡前药拿来"),
        ("check", "living_room", "帮我把灯光调暗一点"),
    ],
    "night": [
        ("fetch", "bedroom", "我有点不舒服，帮我把药箱拿来"),
    ],
}

# 孩子特定请求
CHILD_TEMPLATES = {
    "morning": [
        ("fetch", "bedroom", "帮我把书包拿来"),
        ("fetch", "厨房", "我要吃早饭，帮我去厨房看看"),
    ],
    "afternoon": [
        ("fetch", "bedroom", "帮我把玩具拿来"),
        ("fetch", "living_room", "我想看电视，帮我开电视"),
        ("clean", "bedroom", "卧室乱了，帮忙收拾一下"),
    ],
    "evening": [
        ("fetch", "bedroom", "帮我把故事书拿来"),
        ("fetch", "厨房", "我有点饿，能去厨房拿点吃的吗"),
    ],
    "night": [
        ("check", "bedroom", "帮我看看窗户关好了没有"),
    ],
}


# ---------------------------------------------------------------------------
# 家庭成员类
# ---------------------------------------------------------------------------

class FamilyMember:
    """代表家庭中的一个成员"""

    ROLE_CONFIG = {
        "dad":   {"zh": "爸爸", "color": (0.2, 0.4, 0.8), "height": 1.75, "request_interval": 120},
        "mom":   {"zh": "妈妈", "color": (0.8, 0.3, 0.5), "height": 1.65, "request_interval": 110},
        "child": {"zh": "小孩", "color": (0.3, 0.8, 0.3), "height": 1.20, "request_interval": 90},
        "elder": {"zh": "老人", "color": (0.7, 0.6, 0.3), "height": 1.60, "request_interval": 150},
    }

    def __init__(self, member_id: str, role: str, room_centers: dict,
                 home_room: str = "living_room", index: int = 0):
        self.member_id = member_id
        self.role = role
        self.index = index
        cfg = self.ROLE_CONFIG.get(role, self.ROLE_CONFIG["dad"])
        self.name_zh = cfg["zh"] + (f"{index+1}" if index > 0 else "")
        self.color = cfg["color"]
        self.height = cfg["height"]
        self.request_interval = cfg["request_interval"]   # 虚拟秒 between requests

        self.room_centers = room_centers
        self.home_room = home_room

        # 当前位置（从首选房间取）
        self.position = self._room_pos(home_room)
        self.current_room = home_room

        self._last_request_time = 0.0   # 虚拟时间（秒）
        self.pending_request: Optional[dict] = None
        self._request_countdown = random.uniform(60, 300)   # 首次请求延迟（虚拟s）

        # 活动计划表: [(hour_start, hour_end, room), ...]
        self._schedule = self._build_schedule()

    def _room_pos(self, room_name: str) -> tuple:
        """取房间中心，找不到则返回原点"""
        for key, pos in self.room_centers.items():
            if room_name.lower() in key.lower():
                return pos
        return (0.0, 0.0, 0.1)

    def _build_schedule(self) -> list:
        """根据角色构建一天的活动表"""
        rc = self.room_centers
        kitchen_key  = next((k for k in rc if "kitchen" in k or "厨房" in k), None)
        bedroom_key  = next((k for k in rc if "bedroom" in k), None)
        living_key   = next((k for k in rc if "living" in k), None)
        study_key    = next((k for k in rc if "study" in k), None)
        balcony_key  = next((k for k in rc if "balcony" in k), None)

        schedule = []
        if self.role == "dad":
            schedule = [
                (6,  7,  bedroom_key or "bedroom"),      # 起床
                (7,  8,  kitchen_key or "kitchen"),      # 吃早饭
                (8,  12, study_key or living_key or "study_room"), # 工作
                (12, 13, kitchen_key or "kitchen"),      # 午饭
                (13, 14, bedroom_key or "bedroom"),      # 午休
                (14, 18, study_key or living_key or "study_room"), # 工作
                (18, 19, kitchen_key or "kitchen"),      # 晚饭
                (19, 22, living_key or "living_room"),   # 休息
                (22, 24, bedroom_key or "bedroom"),      # 睡觉
            ]
        elif self.role == "mom":
            schedule = [
                (6,  7,  kitchen_key or "kitchen"),      # 做早饭
                (7,  8,  living_key  or "living_room"),  # 用餐
                (8,  11, living_key  or "living_room"),  # 家务
                (11, 12, kitchen_key or "kitchen"),      # 做午饭
                (12, 13, living_key  or "living_room"),  # 吃午饭
                (13, 15, bedroom_key or "bedroom"),      # 午休
                (15, 17, balcony_key or living_key or "balcony"), # 休闲
                (17, 18, kitchen_key or "kitchen"),      # 做晚饭
                (18, 20, living_key  or "living_room"),  # 吃晚饭/陪家人
                (20, 22, living_key  or "living_room"),  # 看电视
                (22, 24, bedroom_key or "bedroom"),      # 睡觉
            ]
        elif self.role == "child":
            schedule = [
                (7,  8,  kitchen_key or "kitchen"),      # 吃早饭
                (8,  12, study_key   or "study_room"),   # 上课/学习
                (12, 13, kitchen_key or "kitchen"),      # 午饭
                (13, 14, bedroom_key or "bedroom"),      # 午睡
                (14, 17, living_key  or "living_room"),  # 玩耍
                (17, 18, study_key   or "study_room"),   # 做作业
                (18, 19, kitchen_key or "kitchen"),      # 晚饭
                (19, 21, living_key  or "living_room"),  # 玩耍/看电视
                (21, 24, bedroom_key or "bedroom"),      # 睡觉
            ]
        elif self.role == "elder":
            # 老人大多在家，动作慢，休息多
            schedule = [
                (6,  7,  living_key  or "living_room"),  # 早起锻炼
                (7,  8,  kitchen_key or "kitchen"),      # 早饭
                (8,  11, balcony_key or living_key or "balcony"),  # 晒太阳
                (11, 12, living_key  or "living_room"),  # 午前休息
                (12, 13, kitchen_key or "kitchen"),      # 午饭
                (13, 15, bedroom_key or "bedroom"),      # 午睡
                (15, 18, living_key  or "living_room"),  # 休闲
                (18, 19, kitchen_key or "kitchen"),      # 晚饭
                (19, 21, living_key  or "living_room"),  # 看电视
                (21, 24, bedroom_key or "bedroom"),      # 睡觉
            ]
        return schedule

    def update_position(self, current_hour: float):
        """根据时间表更新成员所在房间"""
        for (h_start, h_end, room) in self._schedule:
            if h_start <= current_hour < h_end and room:
                if self.current_room != room:
                    self.current_room = room
                    # 加一点随机偏移避免重叠
                    base = self._room_pos(room)
                    offset = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), 0)
                    self.position = (base[0] + offset[0], base[1] + offset[1], base[2])
                break

    def try_generate_request(self, virtual_seconds: float, period_en: str) -> Optional[dict]:
        """尝试生成一条请求（按间隔）"""
        if virtual_seconds - self._last_request_time < self.request_interval:
            return None

        # 选择模板
        if self.role == "elder":
            templates = ELDER_TEMPLATES.get(period_en, [])
        elif self.role == "child":
            templates = CHILD_TEMPLATES.get(period_en, [])
        else:
            templates = REQUEST_TEMPLATES.get(period_en, [])

        if not templates:
            return None

        # 随机选一条 + 一定概率不发请求（模拟沉默）
        if random.random() < 0.3:   # 30% 概率跳过本次
            self._last_request_time = virtual_seconds
            return None

        action, target_room, text = random.choice(templates)
        request = {
            "requester_id": self.member_id,
            "requester_name": self.name_zh,
            "requester_position": self.position,
            "requester_room": self.current_room,
            "action": action,
            "target_room": target_room,
            "text": text,
            "priority": 1 if self.role == "elder" else 0,  # 老人优先级高
            "virtual_time": virtual_seconds,
        }
        self._last_request_time = virtual_seconds
        return request


# ---------------------------------------------------------------------------
# 家庭模拟器
# ---------------------------------------------------------------------------

class FamilySimulator:
    """
    管理一家人的仿真，计划支持:
      - "nuclear"  : 爸爸 + 妈妈 + 1孩子
      - "large"    : 爸爸 + 妈妈 + 多孩子
      - "elderly"  : 爸爸 + 妈妈 + 老人
      - "full"     : 爸爸 + 妈妈 + 孩子 + 老人
    """

    FAMILY_PRESETS = {
        "nuclear": [("dad", "dad"), ("mom", "mom"), ("child_1", "child")],
        "large":   [("dad", "dad"), ("mom", "mom"),
                    ("child_1", "child"), ("child_2", "child")],
        "elderly": [("dad", "dad"), ("mom", "mom"), ("elder_1", "elder")],
        "full":    [("dad", "dad"), ("mom", "mom"),
                    ("child_1", "child"), ("elder_1", "elder")],
    }

    def __init__(self, rooms_json_path: str, family_type: str = "nuclear"):
        self.room_centers = load_room_centers(rooms_json_path)
        self.members: list[FamilyMember] = []
        self._request_queue: list[dict] = []

        preset = self.FAMILY_PRESETS.get(family_type, self.FAMILY_PRESETS["nuclear"])
        role_counters = {}
        for mid, role in preset:
            role_counters[role] = role_counters.get(role, 0)
            m = FamilyMember(mid, role, self.room_centers, index=role_counters[role])
            role_counters[role] += 1
            self.members.append(m)

        print(f"[FamilySimulator] 家庭类型={family_type}，成员: "
              f"{[m.name_zh for m in self.members]}")

    def step(self, time_manager: TimeManager):
        """每仿真步调用，更新位置并检查请求"""
        h = time_manager.hour
        vs = time_manager.virtual_seconds
        period = time_manager.period_en

        for member in self.members:
            member.update_position(h)
            req = member.try_generate_request(vs, period)
            if req:
                self._request_queue.append(req)
                print(f"  [请求] {req['requester_name']} ({req['requester_room']}): "
                      f"\"{req['text']}\"")

    def pop_requests(self) -> list:
        """取出所有待处理请求并清空队列"""
        reqs = sorted(self._request_queue, key=lambda r: -r["priority"])
        self._request_queue = []
        return reqs

    def get_member_positions(self) -> dict:
        """返回成员ID→位置的字典，用于可视化"""
        return {m.member_id: m.position for m in self.members}

    def get_member_by_id(self, mid: str) -> Optional[FamilyMember]:
        return next((m for m in self.members if m.member_id == mid), None)
