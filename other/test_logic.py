"""
test_logic.py
=============
不依赖 Isaac Sim，仅测试时间管理器、家庭模拟器、机器人大脑逻辑。

用法:  python test_logic.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from time_manager    import TimeManager
from family_simulator import FamilySimulator
from robot_brain     import RobotBrain

ROOMS_JSON = os.path.join(os.path.dirname(__file__),
                          "data", "kujiale_0003", "rooms.json")

def test_simulation(steps: int = 2000, time_scale: float = 720.0):
    print("=" * 60)
    print("  逻辑测试 (无 Isaac Sim)")
    print("=" * 60)

    time_mgr = TimeManager(sim_dt=1.0/120.0, time_scale=time_scale, start_hour=6.0)
    family   = FamilySimulator(ROOMS_JSON, family_type="nuclear")
    brain    = RobotBrain(family.room_centers)

    PRINT_EVERY = 500

    for i in range(steps):
        time_mgr.step()
        family.step(time_mgr)
        new_reqs = family.pop_requests()
        action = brain.update(time_mgr, new_reqs)

        if i % PRINT_EVERY == 0:
            t = time_mgr.summary()
            b = brain.status_summary()
            print(f"\n[帧 {i:4d}] {t['time']} ({t['period_zh']}) | "
                  f"机器人: {b['state']} | 电量: {b['battery']}% | "
                  f"任务: {b['tasks_done']} 完成 | 当前: {b['current_task'] or '—'}")
            print(f"         动作: {action['action']} → {action['description'][:50]}")
            for m in family.members:
                print(f"         {m.name_zh}: 在 {m.current_room}")

    print("\n" + "=" * 60)
    print("  最终统计")
    print(f"  虚拟时间: {time_mgr.time_str}  进度: {time_mgr.progress()*100:.1f}%")
    print(f"  完成任务数: {brain.tasks_completed}")
    print(f"  失败任务数: {brain.tasks_failed}")
    print(f"  最终电量:   {brain.battery:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    test_simulation(steps=3000, time_scale=720.0)
