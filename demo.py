import argparse
import json
import os
import sys
import math
import numpy as np
from isaacsim import SimulationApp
import carb

config = {
    "launch_config": {
        "renderer": "RayTracedLighting",  
        "headless": True,
    },
    "resolution": [1280, 720],
    "writer": "BasicWriter",
}

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=False, action="store_true")
parser.add_argument("--mode", default="manual", help="manual | teleop | policy")
parser.add_argument("--render", default=False, action="store_true")
parser.add_argument("--camera_mode", default="floating")  
parser.add_argument("--camera_height", default=1.5, type=float, help="Camera height from ground")
args, unknown_args = parser.parse_known_args()

config["launch_config"]["headless"] = args.headless
simulation_app = SimulationApp(config["launch_config"])


class FloatingCameraController:
    def __init__(self, world, scene_data):
        self.world = world
        self.state = 0
        from pxr import UsdGeom

        self.camera_height = args.camera_height 
        self.start_position = np.array(scene_data["start_location"]) + np.array([0, 0, self.camera_height])
        self.start_orientation = np.array(scene_data["start_orientation"])

        stage = simulation_app.context.get_stage()
        self.camera_path = "/World/FloatingCamera"
        
        camera_prim = UsdGeom.Camera.Define(stage, self.camera_path)
        camera_prim.CreateFocalLengthAttr(10.0)
        camera_prim.CreateClippingRangeAttr().Set((0.01, 1000.0))
        camera_prim.CreateVerticalApertureAttr(20)
        camera_prim.CreateHorizontalApertureAttr(20.0)

        self.camera = self.camera_path 
        self.current_position = self.start_position.copy()
        self.current_orientation = self.start_orientation.copy()
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.pitch_limit_deg = 85.0 
        self.look_direction = np.array([1, 0, 0]) 

        self.move_speed = 1  
        self.turn_speed = 30.0  

        self.mode = args.mode
        self._base_command = np.array([0.0, 0.0, 0.0])
        self.mission_complete = False 
    
    def reset(self):
        self.current_position = self.start_position.copy()
        self.current_orientation = self.start_orientation.copy()
        self.look_direction = np.array([1, 0, 0])
        self._update_camera()
        self.state = 0
        self.mission_complete = False
        print('=' * 10, "reset", "=" * 10)

    def init_manual(self):
        import omni.appwindow
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._base_command = np.array([0.0, 0.0, 0.0])
        pos_del = 1.0
        ang_del = 1.0
        self._input_keyboard_mapping = {
            "W": [ pos_del, 0.0,   0.0],  # forward
            "S": [-pos_del, 0.0,   0.0],
            "A": [ 0.0,     0.0,    ang_del],  # pitch up
            "D": [ 0.0,     0.0,   -ang_del],  # pitch down
        }

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def rotation_from_direction(self, direction, up_vector=np.array([0, 0, 1])):
        from scipy.spatial.transform import Rotation as R
        direction = np.array(direction, dtype=np.float64)
        forward = direction / np.linalg.norm(direction)
        right = np.cross(up_vector, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1, 0, 0])
        else:
            right = right / right_norm
        up = np.cross(forward, right)
        rot_mat = np.column_stack((forward, right, up))
        quat = R.from_matrix(rot_mat).as_quat()
        return np.array([quat[3], quat[0], quat[1], quat[2]])

    def _update_camera(self):
        from pxr import UsdGeom, Gf
        stage = simulation_app.context.get_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_path)

        if camera_prim and camera_prim.IsValid():
            xformable = UsdGeom.Xformable(camera_prim)
            xformable.ClearXformOpOrder()
            
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(self.current_position[0],
                                     self.current_position[1],
                                     self.current_position[2]))

            orientation = self.rotation_from_direction(self.look_direction)
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])  # xyzw
            euler = r.as_euler('xyz', degrees=True)
            rotate_op = xformable.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(euler[0] + 90, euler[1], euler[2] - 90)) 

    def get_camera_transform(self):
        return {
            "pos_x": self.current_position[0],
            "pos_y": self.current_position[1],
            "pos_z": self.current_position[2],
            "look_x": self.look_direction[0],
            "look_y": self.look_direction[1],
            "look_z": self.look_direction[2],
        }

    def run(self, step_size):
        from scipy.spatial.transform import Rotation as R
        if self.mode == 'manual':
            if abs(self._base_command[0]) > 0:
                self.current_position += self.look_direction * self._base_command[0] * self.move_speed * step_size
            if abs(self._base_command[1]) > 0:
                right_dir = np.cross(self.look_direction, np.array([0, 0, 1]))
                right_dir = right_dir / np.linalg.norm(right_dir)
                self.current_position += right_dir * self._base_command[1] * self.move_speed * step_size
            if abs(self._base_command[2]) > 0:
                angle = self._base_command[2] * self.turn_speed * step_size
                r = R.from_euler('z', angle, degrees=True)
                self.look_direction = r.apply(self.look_direction)
            self.current_position[2] = self.camera_height
        self._update_camera()


reset_needed = False
first_step = True

def run(scene_data):
    from pxr import Sdf
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import define_prim
    import omni.replicator.core as rep

    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=8.0 / 200.0)
    my_world.scene.add_default_ground_plane(z_position=0, name="default_ground_plane", prim_path="/World/defaultGroundPlane")

    if "usd_path" in scene_data:
        prim = define_prim("/World/Ground", "Xform")
        asset_path = scene_data["usd_path"]
        prim.GetReferences().AddReference(asset_path, "/Root")

    camera_controller = FloatingCameraController(world=my_world, scene_data=scene_data)
    if args.mode == "manual":
        camera_controller.init_manual() 

    stage = simulation_app.context.get_stage()
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

    my_world.reset()
    camera_controller.reset()

    global reset_needed
    def on_physics_step(step_size):
        global first_step, reset_needed
        if first_step:
            camera_controller.reset()
            first_step = False
        elif reset_needed:
            my_world.reset(True)
            reset_needed = False
            first_step = True
        else:
            camera_controller.run(step_size)

    my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)
    
    camera_transforms = []
    frame = 0
    mission_complete_signal = False

    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_stopped() and not reset_needed: reset_needed = True
        
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                camera_controller.reset()
                frame = 0
                reset_needed = False
                camera_transforms = []

            if args.render:
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False)

            if camera_controller.mission_complete:
                mission_complete_signal = True
                break

            if args.headless and frame > 2000:
                break
            frame += 1

    if not mission_complete_signal:
        print("\n\n [ABORT] Simulation window closed without pressing ENTER.")
        return None, False

    return None, True

if __name__ == '__main__':
    data_dir = "/home/xunyi/isaacsim4.5/projects/robot_challenge/data"

    scene_id = "kujiale_0003"
    loc = [2.022355307502785, 3.567730767355105, 0.0]
    ori = [0.687852177796288, 0.0, 0.0, -0.7258507983745032]

    usd_path = os.path.join(data_dir, scene_id, f'{scene_id}.usda')

    scene_data = {
        "usd_path": usd_path,
        "start_location": loc,
        "start_orientation": ori,
    }

    run(scene_data)