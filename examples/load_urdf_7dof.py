import argparse
from pathlib import Path
import sys

import numpy as np
import eaik

LOCAL_EAIK_SRC = Path(__file__).resolve().parents[1] / "src" / "eaik"
local_src = str(LOCAL_EAIK_SRC)
if local_src not in eaik.__path__:
    eaik.__path__.insert(0, local_src)

from eaik.IK_URDF import UrdfRobot
from eaik.IK_Redundant import SearchableRedundantUrdfRobot
import evaluate_ik as eval

DEFAULT_URDF = (
    "C:/Users/MichaelBombile/cynpy/azure_codes/robotic-setup-description/"
    "robotic_system/urdf/flexiv_rizon4s.urdf"
)
DEFAULT_BATCH_SIZE = 10
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


def build_7dof_bot(path: str, lock_joint: int, locked_angle: float):
    """Build either a direct analytical 6R reduction or the semi-analytical 7R fallback."""
    if not 1 <= lock_joint <= 7:
        raise ValueError(f"lock_joint must be between 1 and 7, got {lock_joint}")

    lock_index = lock_joint - 1
    bot = UrdfRobot(path, [(lock_index, locked_angle)])
    if bot.hasKnownDecomposition():
        return bot, "analytical", []

    fallback_bot = SearchableRedundantUrdfRobot(path, [(lock_index, locked_angle)])
    if fallback_bot.hasKnownDecomposition():
        return fallback_bot, "semi-analytical", fallback_bot.getSearchJointCandidates()

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
            "The arm also did not reduce to any supported 5R family through the 1D fallback search.\n"
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
    Load a 7-DOF robot from URDF, lock one joint, and run analytical or semi-analytical IK on random poses.
    """
    bot, mode, search_candidates = build_7dof_bot(path, lock_joint, locked_angle)

    print("Kinematic family:", bot.getKinematicFamily())
    print("Spherical wrist:", bot.hasSphericalWrist())
    print("Locked joint:", lock_joint, f"at {locked_angle:.4f} rad")
    print("Solve mode:", mode)
    if search_candidates:
        search_summary = ", ".join(
            f"joint {entry['joint']} -> {entry['family']}" for entry in search_candidates
        )
        print("1D search candidates:", search_summary)

    test_angles = []
    for _ in range(batch_size):
        rand_angles = np.random.random(7) * 2 * np.pi
        rand_angles[lock_joint - 1] = locked_angle
        test_angles.append(rand_angles)

    poses = [bot.fwdKin(angles) for angles in test_angles]

    sum_pos_error = np.array([0.0, 0.0, 0.0])
    sum_rot_error = np.array([0.0, 0.0, 0.0])
    total_num_ls = 0
    for pose_index, pose in enumerate(poses, start=1):
        if mode == "semi-analytical" and batch_size > 1:
            print(f"Solving pose {pose_index}/{batch_size}...")
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
