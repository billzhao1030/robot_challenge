from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MemoryWriter:
    out_dir: str

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    def append(self, agent_name: str, record: Dict[str, Any]) -> None:
        path = os.path.join(self.out_dir, f"{agent_name}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")