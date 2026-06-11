from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot
from eaik.IK_URDF import UrdfRobot, parse_urdf_kinematics


def _family_rank(family: str) -> int:
    upper = family.upper()
    if "UNKNOWN" in upper:
        return 3
    if "PARALLEL" in upper:
        return 2
    return 1


def _config_sort_key(config: dict) -> tuple:
    mode_rank = 0 if config["mode"] == "analytical" else 1
    family_rank = _family_rank(config["family"])
    # Prefer wrist locks (higher joint index) when families tie.
    lock_rank = -config["lock_index"]
    search_rank = config["search_index"] if config["search_index"] is not None else -1
    return (mode_rank, family_rank, lock_rank, search_rank)


def discover_redundancy_configs(
    file_path: str,
    locked_angle: float = 0.0,
    wrist_concurrency_tol: float = -1.0,
) -> list[dict]:
    """
    Enumerate supported single-lock 6R and lock+search 5R reductions for a 7-DOF URDF.

    Returns configs sorted best-first by decomposition quality.
    """
    _, _, H, P, ee_rotation, _ = parse_urdf_kinematics(file_path)
    if H.shape[0] != 7:
        raise ValueError(f"Expected a 7-DOF URDF, got {H.shape[0]} actuated joints.")

    configs: list[dict] = []

    for lock_index in range(7):
        bot = EAIK.Robot(
            H.T,
            P.T,
            ee_rotation,
            [(lock_index, float(locked_angle))],
            True,
            wrist_concurrency_tol,
        )
        if bot.has_known_decomposition():
            configs.append(
                {
                    "mode": "analytical",
                    "lock_joint": lock_index + 1,
                    "lock_index": lock_index,
                    "search_joint": None,
                    "search_index": None,
                    "family": bot.get_kinematic_family(),
                }
            )

    for lock_index in range(7):
        for search_index in range(7):
            if search_index == lock_index:
                continue
            seed = 0.0
            bot = EAIK.Robot(
                H.T,
                P.T,
                ee_rotation,
                sorted([(lock_index, float(locked_angle)), (search_index, seed)]),
                True,
                wrist_concurrency_tol,
            )
            if not bot.has_known_decomposition():
                continue
            family = bot.get_kinematic_family()
            if "Unknown" in family:
                continue
            configs.append(
                {
                    "mode": "semi-analytical",
                    "lock_joint": lock_index + 1,
                    "lock_index": lock_index,
                    "search_joint": search_index + 1,
                    "search_index": search_index,
                    "family": family,
                }
            )

    configs.sort(key=_config_sort_key)
    return configs


def select_redundancy_configs(
    configs: list[dict],
    lock_joint: int | None = None,
    search_joint: int | None = None,
    max_redundancy_configs: int = 1,
    max_search_joint_candidates: int = 1,
) -> list[dict]:
    """Filter and rank redundancy configs for a requested lock/search setup."""
    if not configs:
        return []

    max_redundancy_configs = max(1, int(max_redundancy_configs))
    max_search_joint_candidates = max(1, int(max_search_joint_candidates))

    filtered = configs
    if lock_joint is not None:
        filtered = [c for c in filtered if c["lock_joint"] == lock_joint]
    if search_joint is not None:
        filtered = [c for c in filtered if c.get("search_joint") == search_joint]

    if not filtered:
        return []

    if lock_joint is not None:
        analytical = [c for c in filtered if c["mode"] == "analytical"]
        if analytical:
            return analytical[:1]

        semi = [c for c in filtered if c["mode"] == "semi-analytical"]
        if search_joint is not None:
            return semi[:1]
        return semi[:max_search_joint_candidates]

    return filtered[:max_redundancy_configs]


class ExploringRedundantUrdfRobot(IKRobot):
    """
    Try one or more (lock, search) redundancy reductions and return the best IK solution.
    """

    def __init__(
        self,
        file_path: str,
        configs: list[dict],
        locked_angle: float = 0.0,
        wrist_concurrency_tol: float = -1.0,
        **search_kwargs,
    ):
        if not configs:
            raise ValueError("ExploringRedundantUrdfRobot requires at least one redundancy config.")

        super().__init__()
        self._file_path = file_path
        self._locked_angle = float(locked_angle)
        self._active_configs = list(configs)
        self._solvers: list[IKRobot] = []
        self._solver_labels: list[str] = []

        for config in self._active_configs:
            solver, label = _build_redundancy_solver(
                file_path,
                config,
                locked_angle,
                wrist_concurrency_tol,
                search_joint_candidates=[config["search_index"]]
                if config.get("search_index") is not None
                else None,
                max_search_joint_candidates=1,
                **search_kwargs,
            )
            self._solvers.append(solver)
            self._solver_labels.append(label)

        self._robot = self._solvers[0]._robot

    def getRedundancyConfigs(self) -> list[dict]:
        return list(self._active_configs)

    def getLockJoint(self) -> int:
        return int(self._active_configs[0]["lock_joint"])

    def getSearchJointCandidates(self) -> list[dict]:
        candidates: list[dict] = []
        for config, solver in zip(self._active_configs, self._solvers):
            if config["mode"] != "semi-analytical":
                continue
            if hasattr(solver, "getSearchJointCandidates"):
                candidates.extend(solver.getSearchJointCandidates())
            else:
                candidates.append(
                    {
                        "joint": config["search_joint"],
                        "joint_index": config["search_index"],
                        "family": config["family"],
                        "lock_joint": config["lock_joint"],
                    }
                )
        return candidates

    def getSearchMethod(self) -> str:
        for solver in self._solvers:
            if hasattr(solver, "getSearchMethod"):
                return solver.getSearchMethod()
        return "analytical"

    def hasKnownDecomposition(self) -> bool:
        return any(solver.hasKnownDecomposition() for solver in self._solvers)

    def getKinematicFamily(self) -> str:
        if len(self._active_configs) == 1:
            return self._solvers[0].getKinematicFamily()

        families = []
        for config in self._active_configs:
            if config["mode"] == "analytical":
                families.append(f"lock j{config['lock_joint']}:{config['family']}")
            else:
                families.append(
                    f"lock j{config['lock_joint']}, search j{config['search_joint']}:{config['family']}"
                )
        return "7R-MULTI[" + "; ".join(families) + "]"

    def IK(self, pose: np.ndarray):
        ranked_rows: list[tuple[bool, float, np.ndarray]] = []
        for solver in self._solvers:
            solution = solver.IK(pose)
            if len(solution.Q) == 0:
                continue
            for row, is_ls in zip(np.asarray(solution.Q), np.asarray(solution.is_LS, dtype=bool)):
                row = np.asarray(row, dtype=np.float64)
                err = float(np.linalg.norm(self.fwdKin(row) - pose))
                ranked_rows.append((bool(is_ls), err, row))

        result = EAIK.IKSolution()
        if not ranked_rows:
            return result

        ranked_rows.sort(key=lambda item: (item[0], item[1]))
        rows: list[np.ndarray] = []
        ls_flags: list[bool] = []
        for is_ls, _, row in ranked_rows:
            duplicate = False
            for existing in rows:
                if np.allclose(existing, row, atol=1e-7, rtol=0.0):
                    duplicate = True
                    break
            if duplicate:
                continue
            rows.append(row)
            ls_flags.append(is_ls)
            break

        result.Q = np.vstack(rows)
        result.is_LS = np.asarray(ls_flags, dtype=bool)
        return result

    def IK_batched(self, pose_batch, num_worker_threads=4):
        return [self.IK(pose) for pose in pose_batch]


def _build_redundancy_solver(
    file_path: str,
    config: dict,
    locked_angle: float,
    wrist_concurrency_tol: float,
    search_joint_candidates: list[int] | None = None,
    max_search_joint_candidates: int = 1,
    **search_kwargs,
):
    lock_index = config["lock_index"]
    if config["mode"] == "analytical":
        solver = UrdfRobot(file_path, [(lock_index, locked_angle)], wrist_concurrency_tol)
        label = f"lock j{config['lock_joint']} -> {config['family']}"
        return solver, label

    solver = SearchableRedundantUrdfRobot(
        file_path,
        [(lock_index, locked_angle)],
        search_joint_candidates=search_joint_candidates,
        max_search_joint_candidates=max_search_joint_candidates,
        wrist_concurrency_tol=wrist_concurrency_tol,
        **search_kwargs,
    )
    if search_joint_candidates is not None and len(search_joint_candidates) == 1:
        label = (
            f"lock j{config['lock_joint']}, search j{search_joint_candidates[0] + 1} "
            f"-> {config['family']}"
        )
    else:
        label = f"lock j{config['lock_joint']} -> multi-search"
    return solver, label


def _smoke_test_redundancy_config(
    file_path: str,
    lock_index: int,
    search_index: int,
    locked_angle: float,
    wrist_concurrency_tol: float,
) -> bool:
    """Return True if a lock/search pair survives a single IK smoke test."""
    eaik_src = Path(__file__).resolve().parent
    script = f"""
import numpy as np
import eaik
eaik.__path__.insert(0, {str(eaik_src)!r})
from eaik.IK_Redundant import SearchableRedundantUrdfRobot

bot = SearchableRedundantUrdfRobot(
    {file_path!r},
    [({lock_index}, {float(locked_angle)})],
    search_joint_candidates=[{search_index}],
    search_method="grid",
    search_grid_size=5,
    refinement_steps=0,
    wrist_concurrency_tol={float(wrist_concurrency_tol)},
)
joints = np.zeros(7)
joints[{lock_index}] = {float(locked_angle)}
pose = bot.fwdKin(joints)
solution = bot.IK(pose)
if len(solution.Q) == 0:
    raise SystemExit(2)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return result.returncode == 0


def _best_search_configs_per_lock(
    file_path: str,
    locked_angle: float,
    wrist_concurrency_tol: float,
    max_per_lock: int = 1,
    validate_ik: bool = False,
) -> list[dict]:
    """Rank the best search joint(s) for each lock using kinematic priority."""
    per_lock_best: list[dict] = []
    for lock_index in range(7):
        probe = SearchableRedundantUrdfRobot(
            file_path,
            [(lock_index, locked_angle)],
            max_search_joint_candidates=7,
            wrist_concurrency_tol=wrist_concurrency_tol,
        )
        added = 0
        for entry in probe.getSearchJointCandidates():
            if validate_ik and not _smoke_test_redundancy_config(
                file_path,
                lock_index,
                entry["joint_index"],
                locked_angle,
                wrist_concurrency_tol,
            ):
                continue
            per_lock_best.append(
                {
                    "mode": "semi-analytical",
                    "lock_joint": lock_index + 1,
                    "lock_index": lock_index,
                    "search_joint": entry["joint"],
                    "search_index": entry["joint_index"],
                    "family": entry["family"],
                }
            )
            added += 1
            if added >= max(1, int(max_per_lock)):
                break
    per_lock_best.sort(key=_config_sort_key)
    return per_lock_best


def _auto_select_configs(
    file_path: str,
    locked_angle: float,
    wrist_concurrency_tol: float,
    max_redundancy_configs: int,
) -> list[dict]:
    all_configs = discover_redundancy_configs(file_path, locked_angle, wrist_concurrency_tol)
    analytical = [config for config in all_configs if config["mode"] == "analytical"]
    if analytical:
        return analytical[:1]

    ranked = _best_search_configs_per_lock(
        file_path,
        locked_angle,
        wrist_concurrency_tol,
        max_per_lock=1,
        validate_ik=max_redundancy_configs > 1,
    )
    if not ranked:
        return []
    return ranked[:max(1, int(max_redundancy_configs))]


def build_redundant_urdf_robot(
    file_path: str,
    lock_joint: int | None = None,
    locked_angle: float = 0.0,
    search_joint: int | None = None,
    max_redundancy_configs: int = 1,
    max_search_joint_candidates: int = 1,
    wrist_concurrency_tol: float = -1.0,
    **search_kwargs,
) -> tuple[IKRobot, str, list[dict], int]:
    """
    Build the best available 7-DOF redundancy solver for a URDF.

    Returns (robot, mode, active_configs, resolved_lock_joint).
    """
    if lock_joint is None:
        active_configs = _auto_select_configs(
            file_path,
            locked_angle,
            wrist_concurrency_tol,
            max_redundancy_configs,
        )
        if not active_configs:
            raise ValueError("No supported redundancy configurations found for this URDF.")
        if len(active_configs) == 1:
            lock_joint = active_configs[0]["lock_joint"]
            if active_configs[0]["mode"] == "analytical":
                solver, _ = _build_redundancy_solver(
                    file_path,
                    active_configs[0],
                    locked_angle,
                    wrist_concurrency_tol,
                )
                return solver, "analytical", active_configs, lock_joint
            search_joint = active_configs[0]["search_joint"]
        else:
            solver = ExploringRedundantUrdfRobot(
                file_path,
                active_configs,
                locked_angle=locked_angle,
                wrist_concurrency_tol=wrist_concurrency_tol,
                **search_kwargs,
            )
            return solver, "semi-analytical", solver.getRedundancyConfigs(), solver.getLockJoint()

    lock_index = lock_joint - 1
    analytical_bot = UrdfRobot(file_path, [(lock_index, locked_angle)], wrist_concurrency_tol)
    if analytical_bot.hasKnownDecomposition():
        config = {
            "mode": "analytical",
            "lock_joint": lock_joint,
            "lock_index": lock_index,
            "search_joint": None,
            "search_index": None,
            "family": analytical_bot.getKinematicFamily(),
        }
        return analytical_bot, "analytical", [config], lock_joint

    if search_joint is not None:
        search_indices = [search_joint - 1]
        semi_config = {
            "mode": "semi-analytical",
            "lock_joint": lock_joint,
            "lock_index": lock_index,
            "search_joint": search_joint,
            "search_index": search_joint - 1,
            "family": "",
        }
        solver, _ = _build_redundancy_solver(
            file_path,
            semi_config,
            locked_angle,
            wrist_concurrency_tol,
            search_joint_candidates=search_indices,
            max_search_joint_candidates=1,
            **search_kwargs,
        )
        return solver, "semi-analytical", solver.getSearchJointCandidates(), lock_joint

    solver, _ = _build_redundancy_solver(
        file_path,
        {
            "mode": "semi-analytical",
            "lock_joint": lock_joint,
            "lock_index": lock_index,
            "search_joint": None,
            "search_index": None,
            "family": "",
        },
        locked_angle,
        wrist_concurrency_tol,
        search_joint_candidates=None,
        max_search_joint_candidates=max_search_joint_candidates,
        **search_kwargs,
    )
    if not solver.hasKnownDecomposition():
        raise ValueError(f"No supported redundancy configuration for lock joint {lock_joint}.")
    return solver, "semi-analytical", solver.getSearchJointCandidates(), lock_joint


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
        if search_method not in {"brent", "grid", "analytical"}:
            raise ValueError("search_method must be 'brent', 'grid', or 'analytical'")

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
            try:
                robot = EAIK.Robot(
                    self._H.T,
                    self._P.T,
                    self._R6T,
                    list(key),
                    True,
                    self._wrist_concurrency_tol,
                )
            except RuntimeError:
                return None
            self._robot_cache[key] = robot
        return robot

    def _supports_sp3_analytical(self, family: str) -> bool:
        return "INTERSECTING" in family and "PARALLEL" not in family

    def _sp3_q3_candidates(
        self,
        pose: np.ndarray,
        joint_index: int,
        seed_q3: float,
    ) -> list[float]:
        """
        Closed-form SP3 candidates for the 5R subproblem that normally searches q3.

        Uses the same p_15 construction as EAIK's 5R-FOURTH_FITH_INTERSECTING* solvers.
        """
        bot = self._get_robot(self._fixed_axes + [(joint_index, float(seed_q3))])
        if bot is None or not bot.has_known_decomposition():
            return [float(seed_q3)]

        p_cols = np.asarray(bot.get_remodeled_P())
        if p_cols.shape[1] < 6:
            return [float(seed_q3)]

        rotation = pose[:3, :3]
        p15 = pose[:3, 3] - p_cols[:, 0] - rotation @ p_cols[:, -1]
        side_a = float(np.linalg.norm(p_cols[:, 3]))
        side_b = float(np.linalg.norm(p_cols[:, 2]))
        side_c = float(np.linalg.norm(p15))
        denom = 2.0 * side_a * side_b
        if denom < 1e-12:
            return [float(seed_q3)]

        cos_q = np.clip((side_a * side_a + side_b * side_b - side_c * side_c) / denom, -1.0, 1.0)
        if abs(cos_q) >= 1.0 - 1e-12:
            return [float(seed_q3)]

        offset = float(np.arccos(cos_q))
        return [float(seed_q3 + offset), float(seed_q3 - offset)]

    def _safe_best_ik_at_search_angle(
        self,
        pose: np.ndarray,
        joint_index: int,
        search_angle: float,
    ) -> tuple[float, np.ndarray | None, bool]:
        try:
            return self._best_ik_at_search_angle(pose, joint_index, float(search_angle))
        except RuntimeError:
            return np.inf, None, True

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
            if bot is None or not bot.has_known_decomposition():
                continue
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
        if bot is None or not bot.has_known_decomposition():
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

    def _collect_sp3_candidates(
        self,
        pose: np.ndarray,
        joint_index: int,
    ) -> list[float]:
        lower, upper = self._joint_bounds(joint_index)
        seeds = [0.0]
        if self._warm_start and joint_index in self._last_search_angle:
            seeds.append(float(self._last_search_angle[joint_index]))
        if lower <= 0.0 <= upper:
            seeds.append(0.0)

        candidates: set[float] = set()
        for seed in seeds:
            for angle in self._sp3_q3_candidates(pose, joint_index, seed):
                candidates.add(float(np.clip(angle, lower, upper)))
        return sorted(candidates)

    def _search_analytical(
        self,
        pose: np.ndarray,
        joint_index: int,
        family: str,
    ) -> list[tuple[bool, float, np.ndarray]]:
        if not self._supports_sp3_analytical(family):
            return self._search_brent(pose, joint_index)

        ranked_rows: list[tuple[bool, float, np.ndarray]] = []
        candidates = self._collect_sp3_candidates(pose, joint_index)

        for _ in range(max(1, self._refinement_steps + 1)):
            best_err = np.inf
            best_q = None
            for q3 in candidates:
                err, row, is_ls = self._safe_best_ik_at_search_angle(pose, joint_index, q3)
                if row is None:
                    continue
                ranked_rows.append((is_ls, err, row))
                if err < best_err:
                    best_err = err
                    best_q = float(q3)
                if err <= self._solution_tolerance:
                    self._last_search_angle[joint_index] = float(q3)
                    return ranked_rows

            if best_q is None:
                break

            self._last_search_angle[joint_index] = best_q
            candidates = self._collect_sp3_candidates(pose, joint_index)
            if best_err <= self._solution_tolerance:
                break

        return ranked_rows

    def getLockJoint(self) -> int:
        return self._fixed_axes[0][0] + 1

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
        if self._search_method == "brent":
            search_fn = self._search_brent
        elif self._search_method == "analytical":
            search_fn = None
        else:
            search_fn = self._search_grid

        for candidate in self._candidate_info:
            joint_index = candidate["joint_index"]
            if search_fn is None:
                ranked_rows.extend(
                    self._search_analytical(pose, joint_index, candidate["family"])
                )
            else:
                ranked_rows.extend(search_fn(pose, joint_index))
            if ranked_rows:
                ranked_rows.sort(key=lambda item: (item[0], item[1]))
                if ranked_rows[0][1] <= self._solution_tolerance:
                    break

        return self._pack_solution(ranked_rows)

    def IK_batched(self, pose_batch, num_worker_threads=4):
        return [self.IK(pose) for pose in pose_batch]
