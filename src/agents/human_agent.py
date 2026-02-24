from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random

from world.world_state import WorldState


ROOMS = ["Kitchen", "LivingRoom", "Bedroom", "Bathroom", "Dock"]


@dataclass
class HumanAgent:
    name: str
    request_probability: float = 0.15
    move_probability: float = 0.20

    def update(self, ws: WorldState) -> None:
        """
        Logical-only human simulation. No physics.
        Updates ws.data in-place.
        """
        person = self._find_person(ws)
        if person is None:
            ws.log_event(f"[HumanAgent] {self.name} not found in world_state.")
            return

        # Occasionally move rooms (logical)
        if random.random() < self.move_probability:
            old = person["location"]
            new = random.choice([r for r in ROOMS if r != old])
            person["location"] = new
            person["state"] = "Walking"
            ws.log_event(f"{self.name} moved {old} -> {new}")

        # Occasionally generate a request
        if random.random() < self.request_probability:
            req = self._generate_request(ws, person)
            if req is not None:
                ws.data["requests"].append(req)
                ws.log_event(f"{self.name} generated request: {req}")

    def _find_person(self, ws: WorldState) -> Optional[Dict[str, Any]]:
        for p in ws.data.get("family", []):
            if p.get("name") == self.name:
                return p
        return None

    def _generate_request(self, ws: WorldState, person: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Keep it simple & deterministic enough for debugging.
        # You can replace this with an LLM call that outputs structured JSON.
        options = []

        # If floor dirty, someone might ask to clean
        if ws.data.get("objects", {}).get("Floor", {}).get("dirty", False):
            options.append({"from": self.name, "type": "clean_floor", "room": "LivingRoom"})

        # If cup exists, someone might ask for delivery
        if "Cup_1" in ws.data.get("objects", {}):
            options.append({"from": self.name, "type": "deliver_object", "object": "Cup_1", "to": self.name})

        # Laundry request
        if "Laundry" in ws.data.get("objects", {}):
            options.append({"from": self.name, "type": "handle_laundry", "object": "Laundry"})

        if not options:
            return None

        return random.choice(options)