from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from world.world_state import WorldState
from sim.sim_interface import SimInterface


@dataclass
class RobotExecutor:
    """
    Converts symbolic commands into low-level calls on SimInterface.
    This is where navigation/pick/place/skills live.
    """
    sim: SimInterface

    def execute(self, cmd: Dict[str, Any], ws: WorldState) -> None:
        robot = ws.data["robot"]
        robot["busy"] = True
        robot["last_command"] = cmd
        ws.log_event(f"[Executor] start {cmd}")

        try:
            ctype = cmd.get("type")

            if ctype == "idle":
                # Do nothing, but mark not busy quickly
                ws.log_event("[Executor] idle")
                return

            if ctype == "go_charge":
                self._go_charge(ws)
                return

            if ctype == "deliver":
                self._deliver(cmd, ws)
                return

            if ctype == "clean_floor":
                self._clean_floor(cmd, ws)
                return

            if ctype == "handle_laundry":
                self._handle_laundry(cmd, ws)
                return

            ws.log_event(f"[Executor] Unknown command: {cmd}")

        finally:
            # In a real system, you might keep busy=True until multi-step behaviors finish asynchronously.
            robot["busy"] = False
            ws.log_event(f"[Executor] done {cmd}")

    def _go_charge(self, ws: WorldState) -> None:
        ok = self.sim.dock_and_charge()
        if ok:
            ws.data["robot"]["location"] = "Dock"
            ws.data["robot"]["battery"] = 1.0
            ws.log_event("Robot charged to 100%")
        else:
            ws.log_event("Robot failed to charge")

    def _deliver(self, cmd: Dict[str, Any], ws: WorldState) -> None:
        obj = cmd["object"]
        target_person = cmd["target_person"]

        # Find person location logically
        person_room = None
        for p in ws.data.get("family", []):
            if p.get("name") == target_person:
                person_room = p.get("location")
                break
        if person_room is None:
            ws.log_event(f"Deliver failed: person {target_person} not found")
            return

        # Find object logical location
        obj_info = ws.data.get("objects", {}).get(obj)
        if not obj_info:
            ws.log_event(f"Deliver failed: object {obj} not in world_state")
            return
        obj_room = self._room_from_location(obj_info.get("location", "KitchenTable"))

        # Navigate to object
        self._navigate(ws, obj_room)

        # Pick
        if not self.sim.pick_object(obj):
            ws.log_event(f"Pick failed for {obj}")
            return
        ws.data["robot"]["holding"] = obj
        ws.log_event(f"Picked {obj}")

        # Navigate to person
        self._navigate(ws, person_room)

        # Place
        if not self.sim.place_object(obj, person_room):
            ws.log_event(f"Place failed for {obj}")
            return

        ws.data["robot"]["holding"] = None
        # Update object location in world_state
        ws.data["objects"][obj]["location"] = f"{target_person}_Hands"
        ws.log_event(f"Delivered {obj} to {target_person}")

        # Battery drain (toy)
        ws.data["robot"]["battery"] = max(0.0, ws.data["robot"]["battery"] - 0.05)

    def _clean_floor(self, cmd: Dict[str, Any], ws: WorldState) -> None:
        room = cmd.get("room", "LivingRoom")
        self._navigate(ws, room)

        # In real: trigger cleaning skill (arm + tool)
        floor = ws.data.get("objects", {}).get("Floor", {})
        if floor:
            floor["dirty"] = False
            ws.log_event(f"Cleaned floor in {room}")
        ws.data["robot"]["battery"] = max(0.0, ws.data["robot"]["battery"] - 0.03)

    def _handle_laundry(self, cmd: Dict[str, Any], ws: WorldState) -> None:
        obj = cmd.get("object", "Laundry")
        obj_info = ws.data.get("objects", {}).get(obj)
        if not obj_info:
            ws.log_event(f"Laundry failed: {obj} missing")
            return

        from_room = self._room_from_location(obj_info.get("location", "BedroomFloor"))
        self._navigate(ws, from_room)
        if not self.sim.pick_object(obj):
            ws.log_event(f"Pick failed for {obj}")
            return
        ws.data["robot"]["holding"] = obj

        # Pretend washer is in Bathroom (or LaundryRoom if you add it)
        washer_room = "Bathroom"
        self._navigate(ws, washer_room)

        if not self.sim.place_object(obj, washer_room):
            ws.log_event(f"Place failed for {obj}")
            return

        ws.data["robot"]["holding"] = None
        ws.data["objects"][obj]["location"] = "Washer"
        ws.log_event("Laundry placed into washer")
        ws.data["robot"]["battery"] = max(0.0, ws.data["robot"]["battery"] - 0.06)

    def _navigate(self, ws: WorldState, target_room: str) -> None:
        ok = self.sim.move_robot_base(target_room)
        if ok:
            ws.data["robot"]["location"] = target_room
            ws.log_event(f"Robot navigated to {target_room}")
        else:
            ws.log_event(f"Robot navigation failed to {target_room}")

    @staticmethod
    def _room_from_location(loc: str) -> str:
        # Tiny mapper for toy locations → rooms
        if "Kitchen" in loc:
            return "Kitchen"
        if "Bedroom" in loc:
            return "Bedroom"
        if "LivingRoom" in loc:
            return "LivingRoom"
        if "Bathroom" in loc:
            return "Bathroom"
        if "Dock" in loc:
            return "Dock"
        # Defaults
        return "Kitchen"