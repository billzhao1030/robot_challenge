from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class WaypointController:
    def __init__(
        self,
        robot,
        waypoints_xyz: list[tuple[float, float, float]],
        move_speed: float,
        turn_speed_rad: float,
        position_tolerance: float,
        yaw_tolerance_rad: float,
    ) -> None:
        self.robot = robot
        self.device = robot.device
        self.move_speed = move_speed
        self.turn_speed_rad = turn_speed_rad
        self.position_tolerance = position_tolerance
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.zero_root_velocity = torch.zeros((self.robot.num_instances, 6), device=self.device)
        self.waypoints = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.current_index = 0
        self.finished = True
        self.set_waypoints(waypoints_xyz)

    def set_waypoints(self, waypoints_xyz: list[tuple[float, float, float]]) -> None:
        if len(waypoints_xyz) == 0:
            self.waypoints = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            self.current_index = 0
            self.finished = True
            print("[Waypoint] route cleared.")
            return

        self.waypoints = torch.tensor(waypoints_xyz, dtype=torch.float32, device=self.device)
        self.current_index = 0
        self.finished = False
        print(f"[Waypoint] loaded {len(waypoints_xyz)} waypoint(s).")

    def initialize(self) -> None:
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.write_data_to_sim()
        self.robot.reset()

    def update(self, dt: float) -> None:
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.write_data_to_sim()
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)

        if self.finished:
            return

        current_pose = self.robot.data.root_pose_w.clone()
        current_pos = current_pose[:, :3]
        current_quat = current_pose[:, 3:7]
        waypoint_target = self.waypoints[self.current_index].view(1, 3)
        target_pos = current_pos.clone()
        target_pos[:, :2] = waypoint_target[:, :2]

        delta = target_pos - current_pos
        planar_delta = delta[:, :2]
        planar_distance = torch.linalg.norm(planar_delta, dim=-1)

        _, _, current_yaw = math_utils.euler_xyz_from_quat(current_quat)
        desired_yaw = torch.where(
            planar_distance > 1.0e-6,
            torch.atan2(planar_delta[:, 1], planar_delta[:, 0]),
            current_yaw,
        )
        yaw_error = wrap_to_pi(desired_yaw - current_yaw)

        if torch.all(planar_distance <= self.position_tolerance):
            snapped_pose = current_pose.clone()
            snapped_pose[:, :2] = waypoint_target[:, :2]
            self.robot.write_root_pose_to_sim(snapped_pose)
            print(f"[Waypoint] reached #{self.current_index}: {waypoint_target[0].tolist()}")
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.finished = True
                print("[Waypoint] all waypoints completed.")
            return

        if torch.any((planar_distance > self.position_tolerance) & (torch.abs(yaw_error) > self.yaw_tolerance_rad)):
            yaw_step = torch.clamp(yaw_error, min=-self.turn_speed_rad * dt, max=self.turn_speed_rad * dt)
            next_yaw = current_yaw + yaw_step
            next_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(next_yaw),
                torch.zeros_like(next_yaw),
                next_yaw,
            )
            next_pose = current_pose.clone()
            next_pose[:, 3:7] = next_quat
            self.robot.write_root_pose_to_sim(next_pose)
            return

        step_length = min(self.move_speed * dt, float(planar_distance.max().item()))
        step_scale = step_length / max(float(planar_distance.max().item()), 1.0e-6)
        next_pos = current_pos.clone()
        next_pos[:, :2] = current_pos[:, :2] + planar_delta * step_scale
        next_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(desired_yaw),
            torch.zeros_like(desired_yaw),
            desired_yaw,
        )
        next_pose = current_pose.clone()
        next_pose[:, :2] = next_pos[:, :2]
        next_pose[:, 3:7] = next_quat
        self.robot.write_root_pose_to_sim(next_pose)
