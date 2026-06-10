import numpy as np

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot

class DhRobot(IKRobot):
    """A robot parameterized by standard Denavit-Hartenberg parameters"""

    def __init__(self,
                 dh_alpha: np.ndarray,
                 dh_a: np.ndarray,
                 dh_d: np.ndarray,
                 fixed_axes: list[tuple[int, float]] = None,
                 wrist_concurrency_tol: float = -1.0):
        """
        EAIK Robot parametrized by homogeneous joint transformations

        :param dh_alpha
        :param dh_a
        :param dh_d
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        :param wrist_concurrency_tol: Max distance (m) for wrist axes to be treated as concurrent; negative uses default (1e-4)
        """
        super().__init__()
        if fixed_axes is None:
            fixed_axes = []

        self._robot = EAIK.Robot(dh_alpha, dh_a, dh_d, np.eye(3), fixed_axes, True, wrist_concurrency_tol)
