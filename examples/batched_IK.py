import argparse
import time

import numpy as np
import random
from eaik.IK_URDF import UrdfRobot
import evaluate_ik as eval

DEFAULT_URDF = "Puma560.urdf"
DEFAULT_BATCH_SIZE = 100

def batched_ik_example(path, batch_size):
    """
    Loads spherical-wrist robot from urdf, calculates IK using subproblems and checks the solution for a certian batch size
    """
    bot = UrdfRobot(path)

    # Example desired pose
    test_angles = []
    for i in range(batch_size):
        rand_angles = np.array([random.random(), random.random(), random.random(), random.random(), random.random(), random.random()])
        rand_angles *= 2*np.pi
        test_angles.append(rand_angles)
    poses = []
    for angles in test_angles:
       poses.append(bot.fwdKin(angles))

    num_worker_threads = 4
    t0 = time.perf_counter()
    solutions = bot.IK_batched(poses, num_worker_threads=num_worker_threads)
    ik_batched_elapsed_s = time.perf_counter() - t0

    sum_pos_error = np.array([0.,0.,0.])
    total_num_ls = 0
    for i,ik_solution in enumerate(solutions):
        error_sum_pos, error_sum_rot, is_ls  = eval.evaluate_ik(bot, ik_solution, poses[i], np.eye(3))
        if is_ls:
            # LS solution
            total_num_ls += 1
        sum_pos_error+=error_sum_pos
    print("Avg. Position Error: ", sum_pos_error/len(poses))
    print("Number analytical: ", len(poses)-total_num_ls)
    print("Number LS: ", total_num_ls)
    print("IK_batched batch size: ", len(poses))
    print("IK_batched worker threads: ", num_worker_threads)
    print("IK_batched total time (s): ", ik_batched_elapsed_s)
    print("IK_batched time per pose (ms): ", 1e3 * ik_batched_elapsed_s / len(poses))


def main():
    parser = argparse.ArgumentParser(description="Run batched IK on random poses from a URDF robot.")
    parser.add_argument(
        "--urdf",
        default=DEFAULT_URDF,
        help="Path to the robot URDF file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of random poses to solve",
    )
    args = parser.parse_args()
    batched_ik_example(args.urdf, args.batch_size)


if __name__ == "__main__":
    main()
