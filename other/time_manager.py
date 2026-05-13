"""
TimeManager: 模拟一天时间流逝
- 仿真时间比真实时间快（time_scale可调），例如60倍速则1分钟仿真=1小时真实一天
- 提供当前小时、时间段（早晨/上午/中午/下午/傍晚/夜晚）
- 提供家庭成员的活动状态
"""

import math


class TimeManager:
    """
    将仿真步数映射到时间 (0:00 ~ 24:00)。

    参数
    ----
    sim_dt : float
        每一步仿真时间 (秒)，默认 1/120
    time_scale : float
        时间加速倍率。例如 720 表示用 120s 真实时间走 24h 一天
    start_hour : float
        仿真起始小时，默认 6.0 (早上6点)
    """

    PERIOD_MAP = [
        (6,  8,  "早晨",  "morning"),
        (8,  12, "上午",  "morning"),
        (12, 14, "中午",  "noon"),
        (14, 18, "下午",  "afternoon"),
        (18, 21, "傍晚",  "evening"),
        (21, 24, "夜晚",  "night"),
        (0,  6,  "深夜",  "night"),
    ]

    def __init__(self, sim_dt: float = 1.0 / 120.0,
                 time_scale: float = 720.0,
                 start_hour: float = 6.0):
        self.sim_dt = sim_dt
        self.time_scale = time_scale          # 1 仿真秒 = time_scale 虚拟秒
        self.start_hour = start_hour
        self._elapsed_sim_seconds = 0.0       # 累计仿真秒数
        self._day = 0                          # 第几天（从0开始）

    # ------------------------------------------------------------------
    # 更新（每物理步调用一次）
    # ------------------------------------------------------------------
    def step(self, sim_dt: float | None = None):
        dt = sim_dt if sim_dt is not None else self.sim_dt
        self._elapsed_sim_seconds += dt

    # ------------------------------------------------------------------
    # 当前虚拟时间属性
    # ------------------------------------------------------------------
    @property
    def virtual_seconds(self) -> float:
        """从仿真开始累计的虚拟秒数"""
        return self._elapsed_sim_seconds * self.time_scale

    @property
    def hour(self) -> float:
        """当前虚拟小时 (0..24)"""
        total_hours = self.start_hour + self.virtual_seconds / 3600.0
        return total_hours % 24.0

    @property
    def hour_int(self) -> int:
        return int(self.hour)

    @property
    def minute(self) -> int:
        return int((self.hour % 1) * 60)

    @property
    def time_str(self) -> str:
        """格式化为 HH:MM"""
        return f"{self.hour_int:02d}:{self.minute:02d}"

    @property
    def period_zh(self) -> str:
        """当前时段（中文）"""
        h = self.hour
        for s, e, zh, _ in self.PERIOD_MAP:
            if s <= h < e:
                return zh
        return "深夜"

    @property
    def period_en(self) -> str:
        """当前时段（英文）"""
        h = self.hour
        for s, e, _, en in self.PERIOD_MAP:
            if s <= h < e:
                return en
        return "night"

    def is_daytime(self) -> bool:
        return 7 <= self.hour < 22

    def is_sleeping_time(self) -> bool:
        return self.hour >= 22 or self.hour < 6

    # ------------------------------------------------------------------
    # 进度信息
    # ------------------------------------------------------------------
    def progress(self) -> float:
        """今天已过的比例 0~1"""
        return min(1.0, (self.hour - self.start_hour % 24) / 24.0) if False else \
               min(1.0, self.virtual_seconds / 86400.0)

    def summary(self) -> dict:
        return {
            "time": self.time_str,
            "hour": round(self.hour, 2),
            "period_zh": self.period_zh,
            "period_en": self.period_en,
            "is_daytime": self.is_daytime(),
            "progress": round(self.progress(), 3),
        }

    def __repr__(self):
        return f"<TimeManager {self.time_str} ({self.period_zh})>"
