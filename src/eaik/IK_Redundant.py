from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot
from eaik.IK_URDF import parse_urdf_kinematics


class SearchableRedundantUrdfRobot(IKRobot):
    """
    Semi-analytical fallback for unsupported 7-DOF URDF arms.

    The robot is reduced to a supported 5R family by fixing one additional joint,
    then EAIK's existing analytical 5R solvers are used inside a 1D search.
    """

    def __init__(self,
                 file_path: str,
                 fixed_axes: list[tuple[int, float]],
                 search_joint_candidates: list[int] | None = None,
                 search_method: str = "brent",
                 search_grid_size: int = 21,
                 refinement_grid_size: int = 7,
                 refinement_steps: int = 1,
                 max_search_joint_candidates: int = 1,
                 max_returned_solutions: int = 1,
                 wrist_concurrency_tol: float = -1.0,
                 solution_tolerance: float = 1e-5,
                 brent_xatol: float = 1e-4,
                 warm_start: bool = True,
                 warm_start_span: float = 0.75):
        if len(fixed_axes) != 1:
            raise ValueError("SearchableRedundantUrdfRobot expects exactly one pre-locked joint for a 7-DOF arm.")

        search_method = search_method.lower()
        if search_method not in {"brent", "grid"}:
            raise ValueError("search_method must be 'brent' or 'grid'")

        super().__init__()
        self._file_path = file_path
        self._fixed_axes = sorted((int(i), float(q)) for i, q in fixed_axes)
        self._search_method = search_method
        self._search_grid_size = max(5, int(search_grid_size))
        self._refinement_grid_size = max(5, int(refinement_grid_size))
        self._refinement_steps = max(0, int(refinement_steps))
        self._max_search_joint_candidates = max(1, int(max_search_joint_candidates))
        self._max_returned_solutions = max(1, int(max_returned_solutions))
        self._wrist_concurrency_tol = wrist_concurrency_tol
        self._solution_tolerance = float(solution_tolerance)
        self._brent_xatol = float(brent_xatol)
        self._warm_start = bool(warm_start)
        self._warm_start_span = float(warm_start_span)
        self._last_search_angle: dict[int, float] = {}

        robot, joints, H, P, ee_rotation, joint_limits = parse_urdf_kinematics(file_path)
        if H.shape[0] != 7:
            raise ValueError(f"Expected a 7-DOF URDF, got {H.shape[0]} actuated joints.")

        self._urdf_robot = robot
        self._urdf_joints = joints
        self._joint_limits = joint_limits
        self._H = H
        self._P = P
        self._R6T = ee_rotation
        self._robot = EAIK.Robot(H.T, P.T, ee_rotation, self._fixed_axes, True, wrist_concurrency_tol)
        self._robot_cache: dict[tuple[tuple[int, float], ...], EAIK.Robot] = {}

        self._candidate_info = self._discover_candidates(search_joint_candidates)

    def _robot_cache_key(self, fixed_axes: list[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
        return tuple(sorted((int(i), round(float(q), 12)) for i, q in fixed_axes))

    def _get_robot(self, fixed_axes: list[tuple[int, float]]):
        key = self._robot_cache_key(fixed_axes)
        robot = self._robot_cache.get(key)
        if robot is None:
            robot = EAIK.Robot(self._H.T, self._P.T, self._R6T, list(key), True, self._wrist_concurrency_tol)
            self._robot_cache[key] = robot
        return robot

    def _joint_bounds(self, joint_index: int) -> tuple[float, float]:
        lower, upper = self._joint_limits[joint_index]
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            return -np.pi, np.pi
        return float(lower), float(upper)

    def _candidate_priority(self, candidate: dict) -> tuple[int, float, int]:
        family = candidate["family"]
        parallel_penalty = 1 if "PARALLEL" in family else 0
        midpoint = 0.5 * (self._H.shape[0] - 2)
        middle_distance = abs(candidate["joint_index"] - midpoint)
        return (parallel_penalty, middle_distance, candidate["joint_index"])

    def _discover_candidates(self, requested_candidates: list[int] | None):
        locked_joint = self._fixed_axes[0][0]
        default_candidate_selection = requested_candidates is None
        if requested_candidates is None:
            requested_candidates = [i for i in range(self._H.shape[0]) if i != locked_joint]

        candidate_info = []
        for joint_index in requested_candidates:
            seed = self._seed_angle(joint_index)
            bot = self._get_robot(self._fixed_axes + [(joint_index, seed)])
            if bot.has_known_decomposition():
                candidate_info.append(
                    {
                        "joint_index": joint_index,
                        "joint": joint_index + 1,
                        "seed": seed,
                        "family": bot.get_kinematic_family(),
                    }
                )
        if default_candidate_selection:
            candidate_info.sort(key=self._candidate_priority)
            candidate_info = candidate_info[:self._max_search_joint_candidates]
        return candidate_info

    def _seed_angle(self, joint_index: int) -> float:
        lower, upper = self._joint_bounds(joint_index)
        if lower <= 0.0 <= upper:
            return 0.0
        return 0.5 * (lower + upper)

    def _search_values(self, joint_index: int, center: float | None = None, span: float | None = None):
        lower, upper = self._joint_bounds(joint_index)

        if center is None:
            values = np.linspace(lower, upper, self._search_grid_size)
        else:
            half_span = 0.5 * span if span is not None else 0.5 * (upper - lower)
            values = np.linspace(
                max(lower, center - half_span),
                min(upper, center + half_span),
                self._refinement_grid_size,
            )

        if lower <= 0.0 <= upper:
            values = np.unique(np.concatenate([values, np.array([0.0])]))
        return values

    def _best_ik_at_search_angle(
        self,
        pose: np.ndarray,
        joint_index: int,
        search_angle: float,
    ) -> tuple[float, np.ndarray | None, bool]:
        """Run 5R IK once and score only the best analytical branch."""
        bot = self._get_robot(self._fixed_axes + [(joint_index, float(search_angle))])
        if not bot.has_known_decomposition():
            return np.inf, None, True

        sols = bot.calculate_IK(pose)
        if len(sols.Q) == 0:
            return np.inf, None, True

        best_err = np.inf
        best_row = None
        best_ls = True
        for row, is_ls in zip(np.asarray(sols.Q), np.asarray(sols.is_LS, dtype=bool)):
            row = np.asarray(row, dtype=np.float64)
            err = float(np.linalg.norm(self.fwdKin(row) - pose))
            if err < best_err:
                best_err = err
                best_row = row
                best_ls = bool(is_ls)
        return best_err, best_row, best_ls

    def _append_unique_solution(self, rows, ls_flags, candidate_row, is_ls):
        for i, row in enumerate(rows):
            if np.allclose(row, candidate_row, atol=1e-7, rtol=0.0):
                ls_flags[i] = ls_flags[i] and bool(is_ls)
                return False
        rows.append(candidate_row)
        ls_flags.append(bool(is_ls))
        return True

    def _pack_solution(self, ranked_rows: list[tuple[bool, float, np.ndarray]]) -> EAIK.IKSolution:
        result = EAIK.IKSolution()
        if not ranked_rows:
            return result

        ranked_rows.sort(key=lambda item: (item[0], item[1]))
        ordered_rows: list[np.ndarray] = []
        ordered_ls: list[bool] = []
        for is_ls, _, row in ranked_rows:
            if self._append_unique_solution(ordered_rows, ordered_ls, row, is_ls):
                if len(ordered_rows) >= self._max_returned_solutions:
                    break

        if ordered_rows:
            result.Q = np.vstack(ordered_rows)
            result.is_LS = np.asarray(ordered_ls, dtype=bool)
        return result

    def _search_grid(self, pose: np.ndarray, joint_index: int) -> list[tuple[bool, float, np.ndarray]]:
        lower, upper = self._joint_bounds(joint_index)
        total_span = upper - lower
        ranked_rows: list[tuple[bool, float, np.ndarray]] = []

        values = self._search_values(joint_index)
        local_best_q = None
        local_best_err = np.inf

        for q in values:
            err, row, is_ls = self._best_ik_at_search_angle(pose, joint_index, float(q))
            if row is None:
                continue
            if err < local_best_err:
                local_best_err = err
                local_best_q = float(q)
            ranked_rows.append((is_ls, err, row))
            if err <= self._solution_tolerance:
                self._last_search_angle[joint_index] = float(q)
                return ranked_rows

        if local_best_q is None:
            return ranked_rows

        refine_center = local_best_q
        refine_span = total_span / max(self._search_grid_size - 1, 1)
        for _ in range(self._refinement_steps):
            values = self._search_values(joint_index, refine_center, refine_span)
            for q in values:
                err, row, is_ls = self._best_ik_at_search_angle(pose, joint_index, float(q))
                if row is None:
                    continue
                if err < local_best_err:
                    local_best_err = err
                    refine_center = float(q)
                ranked_rows.append((is_ls, err, row))
                if err <= self._solution_tolerance:
                    self._last_search_angle[joint_index] = float(q)
                    return ranked_rows
            refine_span *= 0.5

        self._last_search_angle[joint_index] = refine_center
        return ranked_rows

    def _brent_bounds(self, joint_index: int) -> tuple[float, float]:
        lower, upper = self._joint_bounds(joint_index)
        if not self._warm_start or joint_index not in self._last_search_angle:
            return lower, upper

        center = self._last_search_angle[joint_index]
        half_span = 0.5 * self._warm_start_span * (upper - lower)
        return max(lower, center - half_span), min(upper, center + half_span)

    def _search_brent(self, pose: np.ndarray, joint_index: int) -> list[tuple[bool, float, np.ndarray]]:
        lower, upper = self._brent_bounds(joint_index)
        state = {
            "best_err": np.inf,
            "best_row": None,
            "best_ls": True,
            "best_q": None,
            "done": False,
        }

        def objective(search_angle: float) -> float:
            if state["done"]:
                return state["best_err"]

            err, row, is_ls = self._best_ik_at_search_angle(pose, joint_index, float(search_angle))
            if row is not None and err < state["best_err"]:
                state["best_err"] = err
                state["best_row"] = row
                state["best_ls"] = is_ls
                state["best_q"] = float(search_angle)

            if state["best_err"] <= self._solution_tolerance:
                state["done"] = True
            return state["best_err"]

        minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": self._brent_xatol},
        )

        if state["best_row"] is None:
            return []

        if state["best_q"] is not None:
            self._last_search_angle[joint_index] = state["best_q"]

        return [(state["best_ls"], state["best_err"], state["best_row"])]

    def getSearchJointCandidates(self) -> list[dict]:
        return list(self._candidate_info)

    def getSearchMethod(self) -> str:
        return self._search_method

    def hasKnownDecomposition(self) -> bool:
        return self._robot.has_known_decomposition() or len(self._candidate_info) > 0

    def getKinematicFamily(self) -> str:
        if self._robot.has_known_decomposition():
            return self._robot.get_kinematic_family()
        if not self._candidate_info:
            return "7R-UNKNOWN"
        families = ", ".join(sorted({entry["family"] for entry in self._candidate_info}))
        return f"7R-SEARCH_OVER_5R[{families}]"

    def IK(self, pose: np.ndarray):
        if self._robot.has_known_decomposition():
            return self._robot.calculate_IK(pose)

        if not self._candidate_info:
            return EAIK.IKSolution()

        ranked_rows: list[tuple[bool, float, np.ndarray]] = []
        search_fn = self._search_brent if self._search_method == "brent" else self._search_grid

        for candidate in self._candidate_info:
            joint_index = candidate["joint_index"]
            ranked_rows.extend(search_fn(pose, joint_index))
            if ranked_rows:
                ranked_rows.sort(key=lambda item: (item[0], item[1]))
                if ranked_rows[0][1] <= self._solution_tolerance:
                    break

        return self._pack_solution(ranked_rows)

    def IK_batched(self, pose_batch, num_worker_threads=4):
        return [self.IK(pose) for pose in pose_batch]
