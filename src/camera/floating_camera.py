import numpy as np
import carb


class FloatingCamera:
    """
    Relative-motion floating camera with explicit yaw/pitch (no accidental roll).

    Keys:
      - W/S: move forward/back (relative to view)
      - A/D: strafe left/right (relative to view)
      - J/L: yaw left/right (look left/right)
      - I/K: pitch up/down (look up/down)
      - R/F: move up/down (world Z)
      - P: reset look up/down to the initial level

    Uses yaw/pitch state -> recompute look_direction each tick.
    """

    def __init__(
        self,
        simulation_app,
        start_location,
        start_orientation,
        camera_height=1.5,
        hfov_deg=90.0,
        vfov_deg=90.0,
        vertical_aperture=20.0,
    ) -> None:
        self.simulation_app = simulation_app

        from pxr import UsdGeom

        stage = self.simulation_app.context.get_stage()

        self.camera_path = "/World/FloatingCamera"
        camera_prim = UsdGeom.Camera.Define(stage, self.camera_path)

        # -------------------------
        # Camera projection setup
        # -------------------------
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.vertical_aperture = float(vertical_aperture)

        self.focal_length = self._focal_from_vfov(
            vertical_aperture=self.vertical_aperture,
            vfov_deg=self.vfov_deg,
        )
        self.horizontal_aperture = self._horizontal_aperture_from_hfov(
            focal_length=self.focal_length,
            hfov_deg=self.hfov_deg,
        )

        camera_prim.CreateFocalLengthAttr(self.focal_length)
        camera_prim.CreateClippingRangeAttr().Set((0.01, 1000.0))
        camera_prim.CreateVerticalApertureAttr(self.vertical_aperture)
        camera_prim.CreateHorizontalApertureAttr(self.horizontal_aperture)

        self.camera_height = float(camera_height)

        self.start_position = np.array(start_location, dtype=np.float64) + np.array(
            [0.0, 0.0, self.camera_height], dtype=np.float64
        )
        self.start_orientation = np.array(start_orientation, dtype=np.float64)  # kept for parity

        self.current_position = self.start_position.copy()

        self.move_speed = 1.0
        self.turn_speed = 30.0

        # Explicit camera angles (degrees)
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.initial_pitch_deg = 0.0

        # Derived forward direction
        self.look_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # [forward, strafe, yaw, pitch, vertical]
        self._base_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        self._appwindow = None
        self._input_keyboard_mapping = {}
        self._viewport_bound = False

    @staticmethod
    def _focal_from_vfov(vertical_aperture: float, vfov_deg: float) -> float:
        vfov_deg = float(np.clip(vfov_deg, 1.0, 179.0))
        return float(vertical_aperture / (2.0 * np.tan(np.deg2rad(vfov_deg) / 2.0)))

    @staticmethod
    def _horizontal_aperture_from_hfov(focal_length: float, hfov_deg: float) -> float:
        hfov_deg = float(np.clip(hfov_deg, 1.0, 179.0))
        return float(2.0 * focal_length * np.tan(np.deg2rad(hfov_deg) / 2.0))

    def apply_fov(self, hfov_deg: float = None, vfov_deg: float = None) -> None:
        """
        Update camera HFOV/VFOV after construction and apply them to the USD camera.
        """
        from pxr import UsdGeom

        if hfov_deg is not None:
            self.hfov_deg = float(hfov_deg)
        if vfov_deg is not None:
            self.vfov_deg = float(vfov_deg)

        self.focal_length = self._focal_from_vfov(
            vertical_aperture=self.vertical_aperture,
            vfov_deg=self.vfov_deg,
        )
        self.horizontal_aperture = self._horizontal_aperture_from_hfov(
            focal_length=self.focal_length,
            hfov_deg=self.hfov_deg,
        )

        stage = self.simulation_app.context.get_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_path)
        if not (camera_prim and camera_prim.IsValid()):
            return

        camera = UsdGeom.Camera(camera_prim)
        camera.GetFocalLengthAttr().Set(self.focal_length)
        camera.GetVerticalApertureAttr().Set(self.vertical_aperture)
        camera.GetHorizontalApertureAttr().Set(self.horizontal_aperture)

    def get_fov(self):
        return {
            "hfov_deg": self.hfov_deg,
            "vfov_deg": self.vfov_deg,
            "focal_length": self.focal_length,
            "horizontal_aperture": self.horizontal_aperture,
            "vertical_aperture": self.vertical_aperture,
        }

    def reset(self) -> None:
        self.current_position = self.start_position.copy()
        self._base_command[:] = 0.0

        self.yaw_deg = 0.0
        self.pitch_deg = self.initial_pitch_deg
        self._recompute_look_direction()
        self._update_camera()

    def init_manual(self) -> None:
        import omni.appwindow

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )

        pos_del = 6.0
        ang_del = 6.0
        up_del = 1.0
        pitch_del = 2.0

        # [forward, strafe, yaw, pitch, vertical]
        self._input_keyboard_mapping = {
            "W": [-pos_del, 0.0, 0.0, 0.0, 0.0],
            "S": [pos_del, 0.0, 0.0, 0.0, 0.0],

            "A": [0.0, -pos_del, 0.0, 0.0, 0.0],
            "D": [0.0, pos_del, 0.0, 0.0, 0.0],

            "J": [0.0, 0.0, ang_del, 0.0, 0.0],   # yaw left
            "L": [0.0, 0.0, -ang_del, 0.0, 0.0],  # yaw right

            "I": [0.0, 0.0, 0.0, -pitch_del, 0.0],  # pitch up
            "K": [0.0, 0.0, 0.0, pitch_del, 0.0],   # pitch down

            "R": [0.0, 0.0, 0.0, 0.0, up_del],    # up
            "F": [0.0, 0.0, 0.0, 0.0, -up_del],   # down
        }
        self._bind_to_active_viewport()

    def _bind_to_active_viewport(self) -> None:
        """Bind the active viewport to this camera so movement is visible."""
        try:
            import omni.kit.viewport.utility as viewport_utils

            viewport = viewport_utils.get_active_viewport()
            if viewport is None:
                return

            if hasattr(viewport, "camera_path"):
                viewport.camera_path = self.camera_path
                self._viewport_bound = True
            elif hasattr(viewport, "set_active_camera"):
                viewport.set_active_camera(self.camera_path)
                self._viewport_bound = True
        except Exception as exc:
            # Keep simulation running even if viewport binding is unavailable.
            carb.log_warn(f"Failed to bind floating camera to active viewport: {exc}")

    def _viewport_has_our_camera(self) -> bool:
        """Check whether active viewport is currently looking through this camera."""
        try:
            import omni.kit.viewport.utility as viewport_utils

            viewport = viewport_utils.get_active_viewport()
            if viewport is None:
                return False

            if hasattr(viewport, "camera_path"):
                return str(viewport.camera_path) == self.camera_path
        except Exception:
            return False
        return False

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "P":
                self.pitch_deg = self.initial_pitch_deg
                self.current_position[2] = self.start_position[2]
                self._recompute_look_direction()
                self._update_camera()
                return True
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(
                    self._input_keyboard_mapping[event.input.name],
                    dtype=np.float64
                )

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(
                    self._input_keyboard_mapping[event.input.name],
                    dtype=np.float64
                )

        return True

    def _recompute_look_direction(self) -> None:
        yaw = np.deg2rad(self.yaw_deg)
        pitch = np.deg2rad(self.pitch_deg)

        cx = np.cos(pitch)
        sx = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        self.look_direction = np.array([cy * cx, sy * cx, sx], dtype=np.float64)
        n = np.linalg.norm(self.look_direction)
        if n > 1e-9:
            self.look_direction /= n

    @staticmethod
    def rotation_from_direction(direction, up_vector=np.array([0.0, 0.0, 1.0], dtype=np.float64)):
        from scipy.spatial.transform import Rotation as R

        direction = np.array(direction, dtype=np.float64)
        n = np.linalg.norm(direction)
        if n < 1e-9:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            n = 1.0
        forward = direction / n

        right = np.cross(up_vector, forward)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / rn

        up = np.cross(forward, right)

        rot_mat = np.column_stack((right, up, forward))
        quat_xyzw = R.from_matrix(rot_mat).as_quat()
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64
        )

    def _update_camera(self) -> None:
        from pxr import UsdGeom, Gf
        from scipy.spatial.transform import Rotation as R

        stage = self.simulation_app.context.get_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_path)
        if not (camera_prim and camera_prim.IsValid()):
            return

        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()

        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(
            float(self.current_position[0]),
            float(self.current_position[1]),
            float(self.current_position[2]),
        ))

        orientation_wxyz = self.rotation_from_direction(self.look_direction)
        r = R.from_quat([
            orientation_wxyz[1],
            orientation_wxyz[2],
            orientation_wxyz[3],
            orientation_wxyz[0],
        ])
        euler = r.as_euler("xyz", degrees=True)

        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(float(euler[0]), float(euler[1]), float(euler[2])))

    def run(self, step_size: float) -> None:
        if (not self._viewport_bound) or (not self._viewport_has_our_camera()):
            self._bind_to_active_viewport()

        forward_cmd, strafe_cmd, yaw_cmd, pitch_cmd, up_cmd = self._base_command

        if abs(yaw_cmd) > 0:
            self.yaw_deg += yaw_cmd * self.turn_speed * step_size

        if abs(pitch_cmd) > 0:
            self.pitch_deg += pitch_cmd * self.turn_speed * step_size

        self.pitch_deg = float(np.clip(self.pitch_deg, -89.0, 89.0))

        self._recompute_look_direction()

        if abs(forward_cmd) > 0:
            self.current_position += self.look_direction * forward_cmd * self.move_speed * step_size

        if abs(strafe_cmd) > 0:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right_dir = np.cross(world_up, self.look_direction)
            n = np.linalg.norm(right_dir)
            if n > 1e-6:
                right_dir = right_dir / n
                self.current_position += right_dir * strafe_cmd * self.move_speed * step_size

        if abs(up_cmd) > 0:
            self.current_position[2] += up_cmd * self.move_speed * step_size

        self._update_camera()
