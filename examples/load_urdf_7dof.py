import argparse
from pathlib import Path
import sys
import time

import numpy as np
import eaik

LOCAL_EAIK_SRC = Path(__file__).resolve().parents[1] / "src" / "eaik"
local_src = str(LOCAL_EAIK_SRC)
if local_src not in eaik.__path__:
    eaik.__path__.insert(0, local_src)

from eaik.IK_URDF import UrdfRobot
from eaik.IK_Redundant import build_redundant_urdf_robot, discover_redundancy_configs
import evaluate_ik as eval

DEFAULT_URDF = (
    "C:/Users/MichaelBombile/cynpy/azure_codes/robotic-setup-description/"
    "robotic_system/urdf/flexiv_rizon4s.urdf"
)
DEFAULT_BATCH_SIZE = 10
DEFAULT_LOCK_JOINT = 7  # 1-based joint index; EAIK example locks the 7th joint (index 6)


class Unsupported7DofRobotError(RuntimeError):
    """Raised when no EAIK subproblem decomposition exists for the locked 7-DOF chain."""


def format_redundancy_table(configs: list[dict]) -> str:
    lines = ["Ranked redundancy configurations:"]
    for index, config in enumerate(configs, start=1):
        if config["mode"] == "analytical":
            lines.append(
                f"  {index:2d}. lock j{config['lock_joint']} -> {config['family']} (analytical)"
            )
        else:
            lines.append(
                f"  {index:2d}. lock j{config['lock_joint']}, search j{config['search_joint']} "
                f"-> {config['family']} (semi-analytical)"
            )
    return "\n".join(lines)


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


def build_7dof_bot(
    path: str,
    lock_joint: int | None,
    locked_angle: float,
    search_options: dict | None = None,
    auto_lock: bool = False,
    search_joint: int | None = None,
):
    """Build the best available analytical or semi-analytical 7-DOF redundancy solver."""
    search_options = dict(search_options or {})
    max_redundancy_configs = int(search_options.pop("max_redundancy_configs", 1))
    max_search_joint_candidates = int(search_options.pop("max_search_joint_candidates", 1))

    if auto_lock:
        lock_joint = None
    elif lock_joint is not None and not 1 <= lock_joint <= 7:
        raise ValueError(f"lock_joint must be between 1 and 7, got {lock_joint}")
    if search_joint is not None and not 1 <= search_joint <= 7:
        raise ValueError(f"search_joint must be between 1 and 7, got {search_joint}")

    try:
        bot, mode, active_configs, resolved_lock_joint = build_redundant_urdf_robot(
            path,
            lock_joint=lock_joint,
            locked_angle=locked_angle,
            search_joint=search_joint,
            max_redundancy_configs=max_redundancy_configs,
            max_search_joint_candidates=max_search_joint_candidates,
            **search_options,
        )
        return bot, mode, active_configs, resolved_lock_joint
    except ValueError as exc:
        candidates = diagnose_lock_candidates(path)
        redundancy = discover_redundancy_configs(path, locked_angle)

        message = str(exc) + "\n"
        if redundancy:
            message += "\n" + format_redundancy_table(redundancy[:15])
            if len(redundancy) > 15:
                message += f"\n  ... and {len(redundancy) - 15} more. Use --list-redundancy to see all."
            message += "\nTry --auto-lock or pass --max-search-candidates > 1.\n"
        else:
            message += "Lock-joint scan:\n"
            for candidate in candidates:
                status = "supported" if candidate["supported"] else "unsupported"
                message += (
                    f"  joint {candidate['joint']}: {candidate['family']} "
                    f"({status}, spherical={candidate['spherical_wrist']})\n"
                )

        raise Unsupported7DofRobotError(message) from exc


def ndof_example(
    path,
    batch_size,
    lock_joint=DEFAULT_LOCK_JOINT,
    locked_angle=0.0,
    measure_time=False,
    search_options: dict | None = None,
    auto_lock: bool = False,
    search_joint: int | None = None,
):
    """
    Load a 7-DOF robot from URDF, lock one joint, and run analytical or semi-analytical IK on random poses.
    """
    bot, mode, search_candidates, resolved_lock_joint = build_7dof_bot(
        path,
        lock_joint,
        locked_angle,
        search_options,
        auto_lock=auto_lock,
        search_joint=search_joint,
    )

    print("Kinematic family:", bot.getKinematicFamily())
    print("Spherical wrist:", bot.hasSphericalWrist())
    if auto_lock:
        print("Lock selection:", "auto")
    print("Locked joint:", resolved_lock_joint, f"at {locked_angle:.4f} rad")
    print("Solve mode:", mode)
    if mode == "semi-analytical" and hasattr(bot, "getSearchMethod"):
        print("1D search method:", bot.getSearchMethod())
    if search_candidates:
        if "search_joint" in search_candidates[0]:
            config_summary = ", ".join(
                f"lock j{entry['lock_joint']}, search j{entry['search_joint']} -> {entry['family']}"
                for entry in search_candidates
            )
            print("Active redundancy configs:", config_summary)
        else:
            search_summary = ", ".join(
                f"joint {entry['joint']} -> {entry['family']}" for entry in search_candidates
            )
            print("1D search candidates:", search_summary)

    test_angles = []
    for _ in range(batch_size):
        rand_angles = np.random.random(7) * 2 * np.pi
        rand_angles[resolved_lock_joint - 1] = locked_angle
        test_angles.append(rand_angles)

    poses = [bot.fwdKin(angles) for angles in test_angles]

    sum_pos_error = np.array([0.0, 0.0, 0.0])
    sum_rot_error = np.array([0.0, 0.0, 0.0])
    total_num_ls = 0
    t0 = time.perf_counter() if measure_time else None
    for pose_index, pose in enumerate(poses, start=1):
        # if mode == "semi-analytical" and batch_size > 1:
        #     print(f"Solving pose {pose_index}/{batch_size}...")
        ik_solution = bot.IK(pose)
        error_sum_pos, error_sum_rot, is_ls = eval.evaluate_ik(bot, ik_solution, pose, np.eye(3))
        if is_ls:
            total_num_ls += 1
        sum_pos_error += error_sum_pos
        sum_rot_error += error_sum_rot
    ik_elapsed_s = time.perf_counter() - t0 if measure_time else None

    print("Avg. Orientation Error:", sum_rot_error / len(poses))
    print("Avg. Position Error:", sum_pos_error / len(poses))
    print("Number analytical:", len(poses) - total_num_ls)
    print("Number LS:", total_num_ls)
    if measure_time:
        print("IK batch size:", len(poses))
        print("IK solve mode:", mode)
        print("IK total time (s):", ik_elapsed_s)
        print("IK time per pose (ms):", 1e3 * ik_elapsed_s / len(poses))


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
        "--auto-lock",
        action="store_true",
        help="Automatically pick the best lock/search redundancy pair for this URDF",
    )
    parser.add_argument(
        "--search-joint",
        type=int,
        default=None,
        help="Optional 1-based joint to search when using semi-analytical fallback",
    )
    parser.add_argument(
        "--max-search-candidates",
        type=int,
        default=1,
        help="When --lock-joint is fixed, try up to this many ranked search joints (default: 1)",
    )
    parser.add_argument(
        "--max-redundancy-configs",
        type=int,
        default=1,
        help="With --auto-lock, try up to this many ranked lock/search pairs per pose (default: 1)",
    )
    parser.add_argument(
        "--list-redundancy",
        action="store_true",
        help="Print ranked lock/search redundancy options for the URDF and exit",
    )
    parser.add_argument(
        "--locked-angle",
        type=float,
        default=0.0,
        help="Fixed angle (rad) for the locked joint",
    )
    parser.add_argument(
        "--measure-time",
        action="store_true",
        help="Measure and print IK loop wall-clock time at the end",
    )
    parser.add_argument(
        "--search-method",
        choices=("analytical", "brent", "grid"),
        default="analytical",
        help="1D redundancy search: analytical (SP3 roots + 5R IK), brent, or grid",
    )
    parser.add_argument(
        "--search-grid-size",
        type=int,
        default=21,
        help="Grid samples for --search-method grid (default: 21)",
    )
    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=1,
        help="Local refinement passes for --search-method grid (default: 1)",
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=1,
        help="Maximum IK solutions returned per pose (default: 1)",
    )
    parser.add_argument(
        "--solution-tolerance",
        type=float,
        default=1e-5,
        help="Stop 1D search early when FK error falls below this (default: 1e-5)",
    )
    parser.add_argument(
        "--brent-xatol",
        type=float,
        default=1e-4,
        help="Angle tolerance for Brent search in radians (default: 1e-4)",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable warm-starting Brent search from the previous pose's best redundancy angle",
    )
    args = parser.parse_args()

    if args.list_redundancy:
        configs = discover_redundancy_configs(args.urdf, args.locked_angle)
        if not configs:
            print("No supported redundancy configurations found.", file=sys.stderr)
            sys.exit(1)
        print(format_redundancy_table(configs))
        sys.exit(0)

    search_options = {
        "search_method": args.search_method,
        "search_grid_size": args.search_grid_size,
        "refinement_steps": args.refinement_steps,
        "max_returned_solutions": args.max_solutions,
        "solution_tolerance": args.solution_tolerance,
        "brent_xatol": args.brent_xatol,
        "warm_start": not args.no_warm_start,
        "max_search_joint_candidates": args.max_search_candidates,
        "max_redundancy_configs": args.max_redundancy_configs,
    }

    try:
        ndof_example(
            args.urdf,
            args.batch_size,
            args.lock_joint,
            args.locked_angle,
            measure_time=args.measure_time,
            search_options=search_options,
            auto_lock=args.auto_lock,
            search_joint=args.search_joint,
        )
    except Unsupported7DofRobotError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
