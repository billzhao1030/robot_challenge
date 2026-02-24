from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time
import random


@dataclass
class RobotPose:
    room: str
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class SimInterface:
    """
    Pure simulation adapter.
    In the MVP, it's a stub. Later, implement the same methods using Isaac Sim APIs.
    """
    def step(self, dt: float) -> None:
        raise NotImplementedError

    def get_robot_pose(self) -> RobotPose:
        raise NotImplementedError

    def get_camera_image(self) -> Any:
        raise NotImplementedError

    def move_robot_base(self, target_room: str) -> bool:
        raise NotImplementedError

    def pick_object(self, object_id: str) -> bool:
        raise NotImplementedError

    def place_object(self, object_id: str, target_room: str) -> bool:
        raise NotImplementedError

    def dock_and_charge(self) -> bool:
        raise NotImplementedError


class SimInterfaceStub(SimInterface):
    """
    A runnable stand-in so you can validate architecture *today*.
    It pretends navigation/pick/place succeed with small delays.
    """
    def __init__(self) -> None:
        self._pose = RobotPose(room="Dock")
        self._last_step_t = time.time()

    def step(self, dt: float) -> None:
        # In Isaac: world.step(render=False) / timeline etc.
        # Here: do nothing.
        time.sleep(0.0)

    def get_robot_pose(self) -> RobotPose:
        return self._pose

    def get_camera_image(self) -> Any:
        # In Isaac: return RGB array.
        # Here: return None.
        return None

    def move_robot_base(self, target_room: str) -> bool:
        # Fake travel time
        time.sleep(0.05)
        self._pose.room = target_room
        return True

    def pick_object(self, object_id: str) -> bool:
        time.sleep(0.03)
        # Random small failure chance for realism
        return random.random() > 0.02

    def place_object(self, object_id: str, target_room: str) -> bool:
        time.sleep(0.03)
        return True

    def dock_and_charge(self) -> bool:
        time.sleep(0.05)
        self._pose.room = "Dock"
        return True