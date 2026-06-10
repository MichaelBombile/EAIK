import argparse
import sys

import numpy as np
import random
from eaik.IK_URDF import UrdfRobot
import evaluate_ik as eval

DEFAULT_URDF = (
    "C:/Users/MichaelBombile/cynpy/azure_codes/robotic-setup-description/"
    "robotic_system/urdf/flexiv_rizon4s.urdf"
)
DEFAULT_BATCH_SIZE = 100
DEFAULT_LOCK_JOINT = 7  # 1-based joint index; EAIK example locks the 7th joint (index 6)


class Unsupported7DofRobotError(RuntimeError):
    """Raised when no EAIK subproblem decomposition exists for the locked 7-DOF chain."""


def diagnose_lock_candidates(path: str) -> list[dict]:
    """Try locking each joint and report whether EAIK recognizes the reduced 6R chain."""
    results = []
    for joint_index in range(7):
        bot = UrdfRobot(path, [(joint_index, 0.0)])
        results.append(
            {
                "joint": joint_index + 1,
                "family": bot.getKinematicFamily(),
                "supported": bot.hasKnownDecomposition(),
                "spherical_wrist": bot.hasSphericalWrist(),
            }
        )
    return results


def ensure_supported_7dof_bot(path: str, lock_joint: int, locked_angle: float) -> UrdfRobot:
    """
    Build a 7-DOF robot with one locked joint and verify EAIK can solve the reduced chain.

    :param lock_joint: 1-based joint index to lock (matches URDF joint numbering)
    :param locked_angle: fixed angle (rad) for the locked joint
    """
    if not 1 <= lock_joint <= 7:
        raise ValueError(f"lock_joint must be between 1 and 7, got {lock_joint}")

    lock_index = lock_joint - 1
    bot = UrdfRobot(path, [(lock_index, locked_angle)])

    if bot.hasKnownDecomposition():
        return bot

    candidates = diagnose_lock_candidates(path)
    supported = [c for c in candidates if c["supported"]]

    message = (
        f"EAIK cannot solve this 7-DOF robot with joint {lock_joint} locked at "
        f"{locked_angle:.4f} rad.\n"
        f"  Kinematic family: {bot.getKinematicFamily()}\n"
        f"  Spherical wrist:  {bot.hasSphericalWrist()}\n"
        f"  Known decomposition: {bot.hasKnownDecomposition()}\n"
    )

    if supported:
        supported_joints = ", ".join(str(c["joint"]) for c in supported)
        message += (
            f"Other lock joints that EAIK does support for this URDF: {supported_joints}\n"
            f"Re-run with --lock-joint set to one of those values.\n"
        )
    else:
        message += (
            "No single-joint lock produced a supported 6R decomposition for this URDF.\n"
            "This arm likely needs a new EAIK kinematic family (for example offset-wrist 7R)\n"
            "or a numerical IK solver such as Flexiv RDK / Pinocchio / TRAC-IK.\n"
        )

    message += "Lock-joint scan:\n"
    for candidate in candidates:
        status = "supported" if candidate["supported"] else "unsupported"
        message += (
            f"  joint {candidate['joint']}: {candidate['family']} "
            f"({status}, spherical={candidate['spherical_wrist']})\n"
        )

    raise Unsupported7DofRobotError(message)


def ndof_example(path, batch_size, lock_joint=DEFAULT_LOCK_JOINT, locked_angle=0.0):
    """
    Load a 7-DOF robot from URDF, lock one joint, and run analytical IK on random poses.
    """
    bot = ensure_supported_7dof_bot(path, lock_joint, locked_angle)

    print("Kinematic family:", bot.getKinematicFamily())
    print("Spherical wrist:", bot.hasSphericalWrist())
    print("Locked joint:", lock_joint, f"at {locked_angle:.4f} rad")

    test_angles = []
    for _ in range(batch_size):
        rand_angles = np.array([random.random()] * 7)
        rand_angles *= 2 * np.pi
        rand_angles[lock_joint - 1] = locked_angle
        test_angles.append(rand_angles)

    poses = [bot.fwdKin(angles) for angles in test_angles]

    sum_pos_error = np.array([0.0, 0.0, 0.0])
    sum_rot_error = np.array([0.0, 0.0, 0.0])
    total_num_ls = 0
    for pose in poses:
        ik_solution = bot.IK(pose)
        error_sum_pos, error_sum_rot, is_ls = eval.evaluate_ik(bot, ik_solution, pose, np.eye(3))
        if is_ls:
            total_num_ls += 1
        sum_pos_error += error_sum_pos
        sum_rot_error += error_sum_rot

    print("Avg. Orientation Error:", sum_rot_error / len(poses))
    print("Avg. Position Error:", sum_pos_error / len(poses))
    print("Number analytical:", len(poses) - total_num_ls)
    print("Number LS:", total_num_ls)


def main():
    parser = argparse.ArgumentParser(
        description="Run analytical IK on a 7-DOF URDF robot with one joint locked."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Path to the robot URDF file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of random poses to solve",
    )
    parser.add_argument(
        "--lock-joint",
        type=int,
        default=DEFAULT_LOCK_JOINT,
        help="1-based joint index to lock before solving IK (Panda example uses 4, KUKA uses 3)",
    )
    parser.add_argument(
        "--locked-angle",
        type=float,
        default=0.0,
        help="Fixed angle (rad) for the locked joint",
    )
    args = parser.parse_args()

    try:
        ndof_example(args.urdf, args.batch_size, args.lock_joint, args.locked_angle)
    except Unsupported7DofRobotError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
