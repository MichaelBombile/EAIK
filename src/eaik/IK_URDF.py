from __future__ import annotations

from urchin import URDF
import numpy as np

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot


def _joint_type(joint) -> str | None:
    return getattr(joint, "joint_type", getattr(joint, "type", None))


def _terminal_fixed_transform(robot, joints, fk_zero_pose):
    """Return the fixed transform from the last actuated link to the terminal link."""
    last_link_name = joints[-1].child
    next_by_parent = {}
    for joint in robot.joints:
        next_by_parent.setdefault(joint.parent, []).append(joint)

    terminal_link_name = last_link_name
    while True:
        outgoing = next_by_parent.get(terminal_link_name, [])
        if len(outgoing) != 1:
            break
        joint = outgoing[0]
        if _joint_type(joint) != "fixed":
            break
        terminal_link_name = joint.child

    last_link = robot.link_map[last_link_name]
    terminal_link = robot.link_map[terminal_link_name]
    terminal_wrt_last = np.linalg.inv(fk_zero_pose[last_link]).dot(fk_zero_pose[terminal_link])
    return terminal_wrt_last[:-1, -1], terminal_wrt_last[:-1, :-1]


def parse_urdf_kinematics(file_path: str):
    """Parse an URDF into EAIK's H/P convention including the fixed tool transform."""
    robot = URDF.load(file_path, lazy_load_meshes=True)
    joints = robot._sort_joints(robot.actuated_joints)
    fk_zero_pose = robot.link_fk()

    parent_p = np.zeros(3, dtype=np.float64)
    H = np.array([], dtype=np.float64).reshape(0, 3)
    P = np.array([], dtype=np.float64).reshape(0, 3)
    joint_limits: list[tuple[float, float]] = []

    for joint in joints:
        joint_child_link = robot.link_map[joint.child]
        h, p = IKRobot.urdf_to_sp_conv(fk_zero_pose[joint_child_link], joint.axis, parent_p)
        H = np.vstack([H, h])
        P = np.vstack([P, p])
        parent_p += p

        lower = getattr(joint.limit, "lower", -np.pi) if getattr(joint, "limit", None) else -np.pi
        upper = getattr(joint.limit, "upper", np.pi) if getattr(joint, "limit", None) else np.pi
        joint_limits.append((float(lower), float(upper)))

    ee_translation, ee_rotation = _terminal_fixed_transform(robot, joints, fk_zero_pose)
    P = np.vstack([P, ee_translation])
    return robot, joints, H, P, ee_rotation, joint_limits


class UrdfRobot(IKRobot):
    """A robot for which the kinematic chain is parsed from a URDF file."""

    def __init__(self,
                 file_path: str,
                 fixed_axes: list[tuple[int, float]] = None,
                 wrist_concurrency_tol: float = -1.0):
        """
        EAIK Robot parametrized by URDF file

        :param file_path: Path to URDF file
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        :param wrist_concurrency_tol: Max distance (m) for wrist axes to be treated as concurrent; negative uses default (1e-4)
        """
        if fixed_axes is None:
            fixed_axes = []
        super().__init__()

        robot, joints, H, P, ee_rotation, joint_limits = parse_urdf_kinematics(file_path)
        self._urdf_robot = robot
        self._urdf_joints = joints
        self._joint_limits = joint_limits
        self._H = H
        self._P = P
        self._R6T = ee_rotation
        self._robot = EAIK.Robot(H.T, P.T, ee_rotation, fixed_axes, True, wrist_concurrency_tol)
