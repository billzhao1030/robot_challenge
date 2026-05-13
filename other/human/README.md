# Three-People Isaac Sim Demo
## What's included

| File | Purpose |
|---|---|
| `three_people_sim.py` | Main executable script |
| `character_commands.txt` | Action sequence for all 3 people |
| `sim_config.yaml` | IRA configuration (scene, camera, output) |
| `README.md` | This file |

---

## Quick-start

### 1 — Copy the folder anywhere on disk
```bash
cp -r three_people_sim/ ~/my_sim/
cd ~/my_sim/
```

### 2 — Run with Isaac Sim's bundled Python

**Linux**
```bash
cd /path/to/isaac-sim-4.5.0
./python.sh ~/my_sim/three_people_sim.py
```

**Windows (PowerShell)**
```powershell
cd C:\path\to\isaac-sim-4.5.0
.\python.bat C:\my_sim\three_people_sim.py
```

### 3 — Optional flags
```bash
# Headless (no viewport window)
./python.sh three_people_sim.py --headless

# Run for 600 frames (~20 s)
./python.sh three_people_sim.py --frames 600
```

---

## Using the built-in IRA scheduler instead
```bash
./python.sh tools/agent_sdg/sdg_scheduler.py -c ~/my_sim/sim_config.yaml
```

---

## What each person does

| Character prim | Behaviour |
|---|---|
| `Character` | Walks a square patrol route → Idle → LookAround → returns to origin |
| `Character_01` | Idle → LookAround → criss-cross walk → LookAround → returns |
| `Character_02` | LookAround → large loop with a mid-point look → Idle → returns |

---

## Editing actions

Open `character_commands.txt` and add / remove lines:

```
# Walk to XYZ and face <degrees>
Character GoTo  5.0  3.0  0  90

# Stand still for 6 seconds
Character Idle 6

# Scan head left/right for 4 seconds
Character LookAround 4

# Sit on a seat prim for 5 seconds (prim must exist in stage)
Character Sit /World/Chair 5
```

Supported commands: **GoTo**, **Idle**, **LookAround**, **Sit**, **Queue/Dequeue**

---

## Troubleshooting

* **Assets not found** – update the `asset_path` fields in `sim_config.yaml`
  to point to your Nucleus server or local Isaac Sim asset cache.
* **Extension missing** – open Extension Manager in Isaac Sim and enable
  `isaacsim.replicator.agent.core` and `omni.anim.people`.
* **No NavMesh** – open **Window → Navigation → NavMesh** and click **Bake**
  after the scene loads.
