"""
RobotBrain: G1机器人行为决策大脑

功能:
1. 处理家庭成员请求（自然语言理解 → 任务规划）
2. 自主巡逻 & 环境感知（无请求时自我决定该做什么）
3. 导航 + 操作（目标点导航）
4. 状态机 (IDLE / NAVIGATING / EXECUTING / CHARGING / PATROLLING)
5. 电量管理（低电量自动充电）

自主决策规则:
- 地板脏 → 清洁
- 电量<20% → 充电
- 有衣服 → 洗衣/晾衣
- 无任务 → 巡逻检查
"""

import random
import numpy as np
from enum import Enum, auto
from time_manager import TimeManager


class RobotState(Enum):
    IDLE        = auto()
    NAVIGATING  = auto()
    EXECUTING   = auto()
    CHARGING    = auto()
    PATROLLING  = auto()
    WAITING     = auto()


class Task:
    def __init__(self, task_type: str, target_pos: tuple,
                 description: str, priority: int = 0,
                 requester: str | None = None):
        self.task_type   = task_type      # fetch / clean / laundry / charge / patrol
        self.target_pos  = target_pos     # (x, y, z)
        self.description = description
        self.priority    = priority
        self.requester   = requester
        self.status      = "pending"       # pending / in_progress / done / failed


class RobotBrain:
    """
    机器人高层决策大脑，不依赖 Isaac Sim 运行，可单独测试。
    Isaac Sim 的 G1 控制器在 demo_home_challenge.py 中对接这里的输出。
    """

    # 充电站固定位置（可根据场景坐标调整）
    CHARGE_STATION_POS = (-5.0, 2.0, 0.1)

    # 巡逻点（会从房间中心动态生成）
    DEFAULT_PATROL_POINTS = [
        (0.0, 0.0, 0.1),
        (-3.0, 2.0, 0.1),
        (2.0, -2.0, 0.1),
    ]

    def __init__(self, room_centers: dict,
                 battery_max: float = 100.0,
                 low_battery_threshold: float = 20.0):
        self.room_centers = room_centers
        self.battery = battery_max
        self.battery_max = battery_max
        self.low_battery_threshold = low_battery_threshold

        # 状态
        self.state = RobotState.IDLE
        self.current_task: Task | None = None
        self.task_queue: list[Task] = []
        self.position = np.array([0.0, 0.0, 0.1])
        self.nav_target: np.ndarray | None = None

        # 巡逻点（从房间中心生成）
        self.patrol_points = self._build_patrol_points()
        self._patrol_idx = 0

        # 统计
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.log: list[str] = []

        # 自主检测计数器（虚拟秒）
        self._last_auto_check = 0.0
        self._auto_check_interval = 300.0   # 每5虚拟分钟自检一次

    # ------------------------------------------------------------------ #
    # 初始化辅助                                                           #
    # ------------------------------------------------------------------ #
    def _build_patrol_points(self) -> list:
        pts = []
        for key, pos in self.room_centers.items():
            if any(room in key for room in ["living", "kitchen", "bedroom",
                                             "balcony", "study"]):
                pts.append(pos)
        return pts if pts else self.DEFAULT_PATROL_POINTS

    # ------------------------------------------------------------------ #
    # 主更新（每若干帧调用一次，step_virtual_seconds: 距上次调用的虚拟秒数）#
    # ------------------------------------------------------------------ #
    def update(self, time_manager: TimeManager, new_requests: list[dict]):
        """
        机器人大脑主循环，返回下一步动作字典：
        {
            "action": "navigate" | "execute" | "charge" | "idle",
            "target_pos": (x, y, z),
            "description": str,
        }
        """
        # 1. 电量消耗
        self._drain_battery(time_manager.sim_dt * time_manager.time_scale)

        # 2. 处理新请求
        for req in new_requests:
            task = self._parse_request(req)
            if task:
                self.task_queue.append(task)

        # 3. 低电量优先充电
        if self.battery < self.low_battery_threshold:
            if self.state != RobotState.CHARGING:
                self._log(f"电量不足 ({self.battery:.0f}%)，前往充电站")
                self._enqueue_priority(Task(
                    "charge", self.CHARGE_STATION_POS,
                    "前往充电站充电", priority=99
                ))

        # 4. 无任务时自主决策
        vs = time_manager.virtual_seconds
        if not self.task_queue and self.state == RobotState.IDLE:
            if vs - self._last_auto_check > self._auto_check_interval:
                self._last_auto_check = vs
                auto_task = self._autonomous_decision(time_manager)
                if auto_task:
                    self.task_queue.append(auto_task)

        # 5. 从队列取任务
        if self.state == RobotState.IDLE and self.task_queue:
            self.task_queue.sort(key=lambda t: -t.priority)
            self.current_task = self.task_queue.pop(0)
            self.current_task.status = "in_progress"
            self.nav_target = np.array(self.current_task.target_pos)
            self.state = RobotState.NAVIGATING
            self._log(f"开始任务: {self.current_task.description}")

        # 6. 导航模拟（实际物理控制在 Isaac Sim 层）
        action = self._tick_navigation()
        return action

    # ------------------------------------------------------------------ #
    # 请求解析                                                             #
    # ------------------------------------------------------------------ #
    def _parse_request(self, req: dict) -> Task | None:
        """将自然语言请求解析为任务"""
        text = req.get("text", "")
        action = req.get("action", "")
        target_room = req.get("target_room", "")
        requester = req.get("requester_name", "")
        priority = req.get("priority", 0)

        # 查找目标位置（先找 target_room，没有就去请求者位置）
        target_pos = self._find_room_pos(target_room)
        if target_pos is None:
            target_pos = req.get("requester_position", self.CHARGE_STATION_POS)

        desc = f"[{requester}请求] {text}"
        task = Task(action, target_pos, desc, priority=priority, requester=requester)
        self._log(f"解析请求: {desc}")
        return task

    def _find_room_pos(self, room_name: str) -> tuple | None:
        """在 room_centers 里模糊匹配房间"""
        if not room_name:
            return None
        for key, pos in self.room_centers.items():
            if room_name.lower() in key.lower() or key.lower() in room_name.lower():
                return pos
        return None

    # ------------------------------------------------------------------ #
    # 自主决策                                                             #
    # ------------------------------------------------------------------ #
    def _autonomous_decision(self, time_manager: TimeManager) -> Task | None:
        """无请求时，机器人自主决定做什么"""
        h = time_manager.hour
        roll = random.random()

        if self.battery < 30:
            pos = self.CHARGE_STATION_POS
            return Task("charge", pos, "自主: 电量偏低，去充电", priority=5)

        if 8 <= h < 10 and roll < 0.4:
            pos = self._find_room_pos("living_room") or self.patrol_points[0]
            return Task("clean", pos, "自主: 早晨打扫客厅", priority=2)

        if 6 <= h < 8 and roll < 0.3:
            pos = self._find_room_pos("kitchen") or self.patrol_points[0]
            return Task("prepare", pos, "自主: 协助准备早饭", priority=2)

        if 20 <= h < 22 and roll < 0.4:
            pos = self._find_room_pos("living_room") or self.patrol_points[0]
            return Task("clean", pos, "自主: 晚间打扫", priority=1)

        if 22 <= h or h < 6:
            pos = self.CHARGE_STATION_POS
            return Task("charge", pos, "自主: 夜间充电待机", priority=3)

        # 默认 - 巡逻
        if self.patrol_points:
            pt = self.patrol_points[self._patrol_idx % len(self.patrol_points)]
            self._patrol_idx += 1
            return Task("patrol", pt, f"自主: 巡逻检查 (点{self._patrol_idx})", priority=0)

        return None

    # ------------------------------------------------------------------ #
    # 导航 tick                                                            #
    # ------------------------------------------------------------------ #
    def _tick_navigation(self) -> dict:
        """模拟导航进度，返回当前动作"""
        if self.state == RobotState.NAVIGATING and self.nav_target is not None:
            dist = np.linalg.norm(self.position - self.nav_target)
            if dist < 0.5:
                # 到达目标 → 执行任务
                self.state = RobotState.EXECUTING
                self._log(f"到达目标，执行: {self.current_task.description if self.current_task else '?'}")
                return {"action": "execute",
                        "target_pos": tuple(self.nav_target),
                        "description": self.current_task.description if self.current_task else ""}
            else:
                return {"action": "navigate",
                        "target_pos": tuple(self.nav_target),
                        "description": f"导航中 (剩余 {dist:.1f}m)"}

        elif self.state == RobotState.EXECUTING:
            # 假设执行完成（实际需要配合操作控制器）
            if self.current_task:
                if self.current_task.task_type == "charge":
                    self.battery = min(self.battery_max, self.battery + 50)
                    self._log(f"充电完成，电量: {self.battery:.0f}%")
                self.current_task.status = "done"
                self.tasks_completed += 1
                self._log(f"任务完成: {self.current_task.description}")
            self.current_task = None
            self.state = RobotState.IDLE
            return {"action": "idle", "target_pos": tuple(self.position), "description": "任务完成，待机"}

        elif self.state == RobotState.CHARGING:
            self.battery = min(self.battery_max, self.battery + 1)
            if self.battery >= self.battery_max:
                self.state = RobotState.IDLE
                self._log("充电完成，准备接受新任务")
            return {"action": "charge", "target_pos": tuple(self.position), "description": "充电中"}

        return {"action": "idle", "target_pos": tuple(self.position), "description": "待机"}

    # ------------------------------------------------------------------ #
    # 工具                                                                 #
    # ------------------------------------------------------------------ #
    def _drain_battery(self, virtual_seconds: float):
        """消耗电量（每虚拟小时耗约5%）"""
        rate = 5.0 / 3600.0   # % per virtual second
        self.battery = max(0.0, self.battery - rate * virtual_seconds)

    def _enqueue_priority(self, task: Task):
        """清除旧充电任务，插入高优先级任务"""
        self.task_queue = [t for t in self.task_queue if t.task_type != "charge"]
        self.task_queue.insert(0, task)

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"  [RobotBrain] {msg}")

    def status_summary(self) -> dict:
        return {
            "state": self.state.name,
            "battery": round(self.battery, 1),
            "queue_len": len(self.task_queue),
            "tasks_done": self.tasks_completed,
            "current_task": self.current_task.description if self.current_task else None,
        }
