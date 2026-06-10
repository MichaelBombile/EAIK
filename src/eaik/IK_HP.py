import numpy as np

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot

class HPRobot(IKRobot):
    """A robot parameterized by by H and P vectors"""

    def __init__(self,
                 H: np.ndarray,
                 P: np.ndarray,
                 fixed_axes: list[tuple[int, float]] = None,
                 wrist_concurrency_tol: float = -1.0):
        """
        EAIK Robot parametrized by H and P vectors

        :param H: (N, 3) numpy array resembling the unit direction vectors of the joints
        :param P: (N+1, 3) numpy array resembling the offsets between the joint axes 
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        :param wrist_concurrency_tol: Max distance (m) for wrist axes to be treated as concurrent; negative uses default (1e-4)
        """
        super().__init__()
        if fixed_axes is None:
            fixed_axes = []

        self._robot = EAIK.Robot(H.T, P.T, np.eye(3), fixed_axes, True, wrist_concurrency_tol)