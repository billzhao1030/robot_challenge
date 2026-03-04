from __future__ import annotations

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def get_world_translation(prim: Usd.Prim) -> Gf.Vec3d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()


def get_world_orientation(prim: Usd.Prim) -> Gf.Quatd:
    return Gf.Quatd(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotationQuat())


def set_prim_world_pose(
    prim: Usd.Prim,
    world_translation: Gf.Vec3d,
    world_orientation: Gf.Quatd,
) -> None:
    parent_prim = prim.GetParent()
    if parent_prim and parent_prim.IsValid() and parent_prim.IsA(UsdGeom.Xformable):
        parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    else:
        parent_world = Gf.Matrix4d(1.0)

    target_world = Gf.Matrix4d(1.0)
    target_world.SetRotate(world_orientation)
    target_world.SetTranslateOnly(world_translation)

    local_matrix = target_world * parent_world.GetInverse()
    local_matrix.Orthonormalize()
    xformable = UsdGeom.Xformable(prim)

    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]

    translate_op = (
        translate_ops[0]
        if translate_ops
        else xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )
    orient_op = (
        orient_ops[0]
        if orient_ops
        else xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )

    translate_op.Set(local_matrix.ExtractTranslation())
    local_quat = local_matrix.ExtractRotationQuat()
    orient_attr = orient_op.GetAttr()
    if orient_attr.GetTypeName() == "quatf":
        orient_op.Set(Gf.Quatf(local_quat))
    else:
        orient_op.Set(Gf.Quatd(local_quat))


def set_prim_local_pose(
    prim: Usd.Prim,
    local_translation: Gf.Vec3d,
    local_orientation: Gf.Quatd,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]

    translate_op = (
        translate_ops[0]
        if translate_ops
        else xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )
    orient_op = (
        orient_ops[0]
        if orient_ops
        else xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    )

    translate_op.Set(local_translation)
    orient_attr = orient_op.GetAttr()
    if orient_attr.GetTypeName() == "quatf":
        orient_op.Set(Gf.Quatf(local_orientation))
    else:
        orient_op.Set(Gf.Quatd(local_orientation))


def vec3d_to_xyz(value: Gf.Vec3d) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def enable_collision(root_prim: Usd.Prim, approximation_type: str = "convexHull") -> None:
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Gprim):
            collision_api = (
                UsdPhysics.CollisionAPI(desc_prim)
                if desc_prim.HasAPI(UsdPhysics.CollisionAPI)
                else UsdPhysics.CollisionAPI.Apply(desc_prim)
            )
            collision_api.CreateCollisionEnabledAttr(True)
            physx_collision_api = (
                PhysxSchema.PhysxCollisionAPI(desc_prim)
                if desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                else PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            )
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        if desc_prim.IsA(UsdGeom.Mesh):
            mesh_collision_api = (
                UsdPhysics.MeshCollisionAPI(desc_prim)
                if desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                else UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            )
            mesh_collision_api.CreateApproximationAttr().Set(approximation_type)


def enable_dynamic_physics(root_prim: Usd.Prim, mass: float = 0.1) -> None:
    enable_collision(root_prim, approximation_type="convexHull")

    rigid_body_api = (
        UsdPhysics.RigidBodyAPI(root_prim)
        if root_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        else UsdPhysics.RigidBodyAPI.Apply(root_prim)
    )
    rigid_body_api.CreateRigidBodyEnabledAttr(True)

    mass_api = (
        UsdPhysics.MassAPI(root_prim)
        if root_prim.HasAPI(UsdPhysics.MassAPI)
        else UsdPhysics.MassAPI.Apply(root_prim)
    )
    mass_api.CreateMassAttr(mass)

    physx_rigid_body_api = (
        PhysxSchema.PhysxRigidBodyAPI(root_prim)
        if root_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        else PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    )
    physx_rigid_body_api.GetDisableGravityAttr().Set(False)
    physx_rigid_body_api.CreateLinearDampingAttr(0.0)
    physx_rigid_body_api.CreateAngularDampingAttr(0.0)


class ProximityGrabber:
    def __init__(
        self,
        stage,
        robot,
        hand_body_name: str,
        object_prim_path: str,
        trigger_distance: float,
        grabbed_object_usd_path: str,
        robot_prim_path: str,
        hand_attach_offset: Gf.Vec3d,
    ) -> None:
        self.stage = stage
        self.robot = robot
        self.object_prim_path = object_prim_path
        self.object_prim = stage.GetPrimAtPath(object_prim_path)
        self.trigger_distance = trigger_distance
        self.is_attached = False
        self.is_released = False
        self._step_count = 0
        self.grabbed_object_usd_path = grabbed_object_usd_path
        self.robot_prim_path = robot_prim_path
        self.hand_attach_offset = hand_attach_offset
        self.attached_object_prim_path: str | None = None
        self.original_world_orientation = get_world_orientation(self.object_prim)

        hand_body_ids, hand_body_names = robot.find_bodies(hand_body_name)
        if len(hand_body_ids) == 0:
            raise RuntimeError(f"Hand body not found: {hand_body_name}")
        self.hand_body_idx = hand_body_ids[0]
        self.hand_body_name = hand_body_names[0]

        if not self.object_prim or not self.object_prim.IsValid():
            raise RuntimeError(f"Object prim not found: {object_prim_path}")

    def distance_to_object(self) -> float:
        robot_root = self.robot.data.root_pos_w[0].clone()
        object_pos = get_world_translation(self.object_prim)
        return (
            (float(robot_root[0]) - object_pos[0]) ** 2
            + (float(robot_root[1]) - object_pos[1]) ** 2
        ) ** 0.5

    def _attach_object_under_hand(self) -> None:
        object_name = self.object_prim.GetName()
        hand_parent_path = f"{self.robot_prim_path}/{self.hand_body_name}"
        attached_object_path = f"{hand_parent_path}/{object_name}"

        self.stage.RemovePrim(attached_object_path)
        original_override = self.stage.OverridePrim(self.object_prim_path)
        original_override.SetActive(False)

        attached_prim = self.stage.DefinePrim(attached_object_path, "Xform")
        attached_prim.GetReferences().AddReference(self.grabbed_object_usd_path)
        set_prim_local_pose(
            attached_prim,
            local_translation=self.hand_attach_offset,
            local_orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        )

        self.object_prim = attached_prim
        self.attached_object_prim_path = attached_object_path

    def release_to_world(self, release_xyz: tuple[float, float, float]) -> None:
        if not self.is_attached or self.is_released:
            return

        if self.attached_object_prim_path is not None:
            self.stage.RemovePrim(self.attached_object_prim_path)

        restored_prim = self.stage.OverridePrim(self.object_prim_path)
        restored_prim.SetActive(True)
        references = restored_prim.GetReferences()
        references.ClearReferences()
        references.AddReference(self.grabbed_object_usd_path)
        set_prim_world_pose(
            restored_prim,
            world_translation=Gf.Vec3d(*release_xyz),
            world_orientation=self.original_world_orientation,
        )

        self.object_prim = restored_prim
        self.is_released = True
        print(f"[Grab] released {self.object_prim_path} to {list(release_xyz)}")

    def update(self) -> None:
        self._step_count += 1
        if not self.is_attached and not self.is_released:
            distance = self.distance_to_object()
            if self._step_count % 120 == 0:
                print(f"[Grab] distance to object: {distance:.3f} m")
            if distance < self.trigger_distance:
                self.is_attached = True
                self._attach_object_under_hand()
                print(
                    f"[Grab] attached {self.object_prim.GetName()} under "
                    f"{self.robot_prim_path}/{self.hand_body_name} at distance {distance:.3f} m"
                )
