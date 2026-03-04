from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG


@configclass
class HomeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    g1: ArticulationCfg = G1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-5.51, -3.11, 0.74),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 暂时不在房间里放人，先保留原始圆柱体配置但注释掉，不删除。
    #
    # human_0 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Humans/Human_0",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.18,
    #         height=1.75,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         activate_contact_sensors=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 0.0, 0.875]),
    # )
    #
    # human_1 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Humans/Human_1",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.17,
    #         height=1.62,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         activate_contact_sensors=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 1.0, 0.81]),
    # )
    #
    # human_2 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Humans/Human_2",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.14,
    #         height=1.10,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         activate_contact_sensors=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.5, 0.55]),
    # )
