from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--move-speed", type=float, default=1, help="Root translation speed in m/s.")
    parser.add_argument("--turn-speed-deg", type=float, default=360.0, help="Yaw rotation speed in deg/s.")
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.05,
        help="Waypoint reach tolerance in meters.",
    )
    parser.add_argument(
        "--yaw-tolerance-deg",
        type=float,
        default=5.0,
        help="Allowed yaw error before starting translation.",
    )
    parser.add_argument("--map-cell-size", type=float, default=0.05, help="Occupancy map resolution in meters.")
    parser.add_argument(
        "--map-margin",
        type=float,
        default=0.75,
        help="Extra planning margin added around the house bounds in meters.",
    )
    parser.add_argument(
        "--path-clearance",
        type=float,
        default=0.25,
        help="Obstacle inflation radius for planning in meters.",
    )
    parser.add_argument(
        "--plan-debug-vis",
        action="store_true",
        default=True,
        help="Draw occupancy map and planned path into the stage.",
    )
    parser.add_argument(
        "--freemap-path",
        type=str,
        default="/home/xunyi/isaacsim5.1/projects/robot_challenge/data/kujiale_0003/freemap.npy",
        help="Path to the precomputed occupancy freemap .npy file.",
    )
    parser.add_argument(
        "--walk-policy-path",
        type=str,
        default="",
        help="TorchScript locomotion policy path for G1. If empty, falls back to root-motion control.",
    )
    parser.add_argument(
        "--policy-decimation",
        type=int,
        default=4,
        help="Number of physics steps per low-level locomotion policy inference.",
    )
    parser.add_argument(
        "--policy-action-scale",
        type=float,
        default=0.5,
        help="Scale applied to locomotion policy outputs before adding the default joint pose.",
    )
    parser.add_argument(
        "--walk-speed",
        type=float,
        default=0.5,
        help="Target forward walking speed in m/s when using the locomotion policy.",
    )
    parser.add_argument(
        "--walk-yaw-rate",
        type=float,
        default=1.0,
        help="Maximum yaw rate command in rad/s for locomotion-policy walking.",
    )
    parser.add_argument(
        "--walk-heading-gain",
        type=float,
        default=1.8,
        help="Heading controller gain for locomotion-policy walking.",
    )
    parser.add_argument(
        "--walk-heading-stop-deg",
        type=float,
        default=35.0,
        help="Stop forward motion when heading error exceeds this angle in walking mode.",
    )
    return parser.parse_args()
