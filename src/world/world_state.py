from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy
import time


@dataclass
class WorldState:
    """
    Single source of truth (JSON-like) for the entire system.
    Keep it serializable and easy to log.
    """
    data: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def default() -> "WorldState":
        return WorldState(
            data={
                "time_sec": 0,  # simulation logical seconds since start
                "time_str": "00:00",
                "robot": {
                    "location": "Dock",
                    "battery": 1.0,
                    "holding": None,
                    "busy": False,
                    "last_command": None,
                },
                "family": [
                    {"name": "Dad", "location": "Kitchen", "state": "Cooking"},
                    {"name": "Mom", "location": "Bedroom", "state": "Resting"},
                    {"name": "Child", "location": "LivingRoom", "state": "Playing"},
                ],
                "objects": {
                    "Cup_1": {"location": "KitchenTable"},
                    "Laundry": {"location": "BedroomFloor"},
                    "Floor": {"dirty": True, "location": "LivingRoom"},
                },
                "requests": [],  # list of {"from": "Dad", "type": "...", ...}
                "events": [],    # debug/event log
            }
        )

    def clone(self) -> "WorldState":
        return WorldState(data=copy.deepcopy(self.data))

    def log_event(self, msg: str) -> None:
        self.data.setdefault("events", []).append(
            {"t": self.data.get("time_str", "??:??"), "msg": msg}
        )

    def advance_time(self, dt_sec: int) -> None:
        self.data["time_sec"] += dt_sec
        # Format HH:MM for readability (not tied to wall-clock)
        total_min = self.data["time_sec"] // 60
        hh = (total_min // 60) % 24
        mm = total_min % 60
        self.data["time_str"] = f"{hh:02d}:{mm:02d}"