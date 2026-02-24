from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from world.world_state import WorldState


@dataclass
class RobotBrain:
    """
    High-level decision making. No physics, no Isaac calls.
    Output: a symbolic command dict or None.
    """
    low_battery_threshold: float = 0.25

    def decide(self, ws: WorldState) -> Optional[Dict[str, Any]]:
        robot = ws.data["robot"]

        # If robot already busy, don't issue a new command (executor is still running)
        if robot.get("busy", False):
            return None

        # Battery management first
        if robot["battery"] < self.low_battery_threshold:
            return {"type": "go_charge"}

        # Serve pending requests in FIFO order
        reqs: List[Dict[str, Any]] = ws.data.get("requests", [])
        if reqs:
            req = reqs.pop(0)  # consume the request
            cmd = self._request_to_command(req, ws)
            if cmd is not None:
                return cmd

        # Proactive behaviors (if no request)
        floor = ws.data.get("objects", {}).get("Floor", {})
        if floor.get("dirty", False):
            return {"type": "clean_floor", "room": floor.get("location", "LivingRoom")}

        # Idle default
        return {"type": "idle"}

    def _request_to_command(self, req: Dict[str, Any], ws: WorldState) -> Optional[Dict[str, Any]]:
        t = req.get("type")
        if t == "deliver_object":
            obj = req.get("object")
            to = req.get("to")
            if obj and to:
                return {"type": "deliver", "object": obj, "target_person": to}
        if t == "clean_floor":
            return {"type": "clean_floor", "room": req.get("room", "LivingRoom")}
        if t == "handle_laundry":
            return {"type": "handle_laundry", "object": req.get("object", "Laundry")}
        return None