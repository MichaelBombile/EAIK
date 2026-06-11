#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "IKS.h"
#include "kinematic_remodeling.h"

namespace IKS
{
    namespace
    {
        constexpr double kPi = 3.141592653589793238462643383279502884;

        bool family_is_supported(const std::string &family)
        {
            return family.find("Unknown") == std::string::npos;
        }

        int candidate_priority(const std::string &family, int joint_index)
        {
            const int parallel_penalty = family.find("PARALLEL") != std::string::npos ? 1 : 0;
            const int middle_distance = std::abs(joint_index - 2);
            return parallel_penalty * 100 + middle_distance * 10 + joint_index;
        }
    }

    General_7R::General_7R(const Eigen::MatrixXd &H,
                           const Eigen::MatrixXd &P,
                           const Eigen::Matrix<double, 3, 3> &R6T,
                           int locked_joint_index,
                           double locked_joint_value,
                           double zero_threshold,
                           double axis_intersect_threshold,
                           int search_joint_index)
        : General_Robot(H, P),
          H(H),
          P(P),
          R6T(R6T),
          locked_joint_index(locked_joint_index),
          locked_joint_value(locked_joint_value),
          search_joint_index(search_joint_index),
          ZERO_THRESHOLD(zero_threshold),
          AXIS_INTERSECT_THRESHOLD(axis_intersect_threshold)
    {
        if (H.rows() != 3 || P.rows() != 3 || H.cols() != 7 || P.cols() != 8)
        {
            throw std::runtime_error("General_7R expects H to be 3x7 and P to be 3x8.");
        }
        if (locked_joint_index < 0 || locked_joint_index >= H.cols())
        {
            throw std::runtime_error("General_7R: locked_joint_index out of range.");
        }

        if (this->search_joint_index < 0)
        {
            this->search_joint_index = detect_search_joint();
        }

        if (this->search_joint_index < 0)
        {
            five_r_class = FiveRClass::UNSUPPORTED;
            five_r_family = "7R-UNKNOWN";
            return;
        }

        Eigen::Matrix<double, 3, 5> H5;
        Eigen::Matrix<double, 3, 6> P5;
        Eigen::Matrix<double, 3, 3> R5;
        if (!build_reduced_5r(0.0, H5, P5, R5))
        {
            five_r_class = FiveRClass::UNSUPPORTED;
            five_r_family = "7R-UNKNOWN";
            return;
        }

        General_5R probe(H5, P5);
        five_r_family = probe.get_kinematic_family();
        five_r_class = classify_five_r_family(five_r_family);
    }

    General_7R::FiveRClass General_7R::classify_five_r_family(const std::string &family) const
    {
        if (!family_is_supported(family))
        {
            return FiveRClass::UNSUPPORTED;
        }

        if (family.find("INTERSECTING") != std::string::npos && family.find("PARALLEL") == std::string::npos)
        {
            return FiveRClass::SP3_ANALYTICAL;
        }

        return FiveRClass::GENERIC;
    }

    int General_7R::detect_search_joint()
    {
        struct Candidate
        {
            int joint_index;
            int priority;
            std::string family;
        };

        std::vector<Candidate> candidates;
        for (int joint_index = 0; joint_index < H.cols(); ++joint_index)
        {
            if (joint_index == locked_joint_index)
            {
                continue;
            }

            Eigen::Matrix<double, 3, 5> H5;
            Eigen::Matrix<double, 3, 6> P5;
            Eigen::Matrix<double, 3, 3> R5;
            if (!build_reduced_5r_for_joint(joint_index, 0.0, H5, P5, R5))
            {
                continue;
            }

            General_5R probe(H5, P5);
            const std::string family = probe.get_kinematic_family();
            if (!family_is_supported(family))
            {
                continue;
            }

            candidates.push_back({joint_index, candidate_priority(family, joint_index), family});
        }

        if (candidates.empty())
        {
            return -1;
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
            return a.priority < b.priority;
        });

        return candidates.front().joint_index;
    }

    bool General_7R::build_reduced_5r(double search_angle,
                                      Eigen::Matrix<double, 3, 5> &H5,
                                      Eigen::Matrix<double, 3, 6> &P5,
                                      Eigen::Matrix<double, 3, 3> &R5) const
    {
        return build_reduced_5r_for_joint(search_joint_index, search_angle, H5, P5, R5);
    }

    bool General_7R::build_reduced_5r_for_joint(int search_index,
                                                 double search_angle,
                                                 Eigen::Matrix<double, 3, 5> &H5,
                                                 Eigen::Matrix<double, 3, 6> &P5,
                                                 Eigen::Matrix<double, 3, 3> &R5) const
    {
        try
        {
            std::vector<std::pair<int, double>> fixed_axes = {
                {locked_joint_index, locked_joint_value},
                {search_index, search_angle},
            };

            const auto &[H_part, P_part, R_part] =
                EAIK::partial_joint_parametrization(H, P, fixed_axes, R6T);
            if (H_part.cols() != 5 || P_part.cols() != 6)
            {
                return false;
            }

            const Eigen::MatrixXd P_remodeled =
                EAIK::remodel_kinematics(H_part, P_part, ZERO_THRESHOLD, AXIS_INTERSECT_THRESHOLD);
            H5 = H_part;
            P5 = P_remodeled;
            R5 = R_part;
            return true;
        }
        catch (const std::runtime_error &)
        {
            return false;
        }
    }

    std::vector<double> General_7R::sp3_search_candidates(const Homogeneous_T &ee_pose, double seed_q) const
    {
        std::vector<double> candidates;
        candidates.push_back(seed_q);

        Eigen::Matrix<double, 3, 5> H5;
        Eigen::Matrix<double, 3, 6> P5;
        Eigen::Matrix<double, 3, 3> R5;
        if (!build_reduced_5r(seed_q, H5, P5, R5))
        {
            return candidates;
        }

        const Eigen::Vector3d p15 =
            ee_pose.block<3, 1>(0, 3) - P5.col(0) - ee_pose.block<3, 3>(0, 0) * P5.col(5);
        const double side_a = P5.col(3).norm();
        const double side_b = P5.col(2).norm();
        const double side_c = p15.norm();
        const double denom = 2.0 * side_a * side_b;
        if (denom < ZERO_THRESHOLD)
        {
            return candidates;
        }

        const double cos_q =
            std::clamp((side_a * side_a + side_b * side_b - side_c * side_c) / denom, -1.0, 1.0);
        if (std::abs(cos_q) >= 1.0 - ZERO_THRESHOLD)
        {
            return candidates;
        }

        const double offset = std::acos(cos_q);
        candidates.push_back(seed_q + offset);
        candidates.push_back(seed_q - offset);
        return candidates;
    }

    std::vector<double> General_7R::expand_to_7dof(const std::vector<double> &q5, double search_angle) const
    {
        std::vector<double> q7(H.cols(), 0.0);
        q7.at(static_cast<std::size_t>(locked_joint_index)) = locked_joint_value;
        q7.at(static_cast<std::size_t>(search_joint_index)) = search_angle;

        std::size_t reduced_index = 0;
        for (int joint_index = 0; joint_index < H.cols(); ++joint_index)
        {
            if (joint_index == locked_joint_index || joint_index == search_joint_index)
            {
                continue;
            }
            q7.at(static_cast<std::size_t>(joint_index)) = q5.at(reduced_index++);
        }
        return q7;
    }

    IK_Solution General_7R::solve_at_search_angle(const Homogeneous_T &ee_pose, double search_angle) const
    {
        IK_Solution merged;
        Eigen::Matrix<double, 3, 5> H5;
        Eigen::Matrix<double, 3, 6> P5;
        Eigen::Matrix<double, 3, 3> R5;
        if (!build_reduced_5r(search_angle, H5, P5, R5))
        {
            return merged;
        }

        General_5R reduced_solver(H5, P5);
        if (!family_is_supported(reduced_solver.get_kinematic_family()))
        {
            return merged;
        }

        Homogeneous_T adjusted_pose = ee_pose;
        adjusted_pose.block<3, 3>(0, 0) *= R5.transpose();
        const IK_Solution reduced_solution = reduced_solver.calculate_IK(adjusted_pose);

        for (std::size_t i = 0; i < reduced_solution.Q.size(); ++i)
        {
            merged.Q.push_back(expand_to_7dof(reduced_solution.Q.at(i), search_angle));
            merged.is_LS_vec.push_back(reduced_solution.is_LS_vec.at(i));
        }
        return merged;
    }

    IK_Solution General_7R::rank_solutions_by_fk_error(IK_Solution solution,
                                                       const Homogeneous_T &desired_pose) const
    {
        if (solution.Q.size() <= 1)
        {
            return solution;
        }

        struct RankedSolution
        {
            std::size_t index;
            double error;
            bool is_ls;
        };

        std::vector<RankedSolution> ranked;
        ranked.reserve(solution.Q.size());
        for (std::size_t i = 0; i < solution.Q.size(); ++i)
        {
            const Homogeneous_T result = fwdkin(solution.Q.at(i));
            ranked.push_back({i, (result - desired_pose).norm(), solution.is_LS_vec.at(i)});
        }

        std::sort(ranked.begin(), ranked.end(), [](const RankedSolution &a, const RankedSolution &b) {
            if (a.is_ls != b.is_ls)
            {
                return a.is_ls < b.is_ls;
            }
            return a.error < b.error;
        });

        IK_Solution ordered;
        ordered.Q.reserve(solution.Q.size());
        ordered.is_LS_vec.reserve(solution.is_LS_vec.size());
        for (const RankedSolution &entry : ranked)
        {
            ordered.Q.push_back(solution.Q.at(entry.index));
            ordered.is_LS_vec.push_back(solution.is_LS_vec.at(entry.index));
        }
        return ordered;
    }

    IK_Solution General_7R::calculate_IK(const Homogeneous_T &ee_position_orientation) const
    {
        IK_Solution merged;
        if (!has_known_decomposition())
        {
            return merged;
        }

        std::vector<double> candidates;
        if (five_r_class == FiveRClass::SP3_ANALYTICAL)
        {
            candidates = sp3_search_candidates(ee_position_orientation, 0.0);
        }
        else
        {
            const int grid_size = 21;
            for (int i = 0; i < grid_size; ++i)
            {
                const double angle =
                    -kPi + (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(grid_size - 1);
                candidates.push_back(angle);
            }
        }

        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end(), [](double a, double b) {
                             return std::abs(a - b) < 1e-8;
                         }),
                         candidates.end());

        for (const double search_angle : candidates)
        {
            const IK_Solution branch = solve_at_search_angle(ee_position_orientation, search_angle);
            merged.Q.insert(merged.Q.end(), branch.Q.begin(), branch.Q.end());
            merged.is_LS_vec.insert(merged.is_LS_vec.end(), branch.is_LS_vec.begin(), branch.is_LS_vec.end());
        }

        return rank_solutions_by_fk_error(
            enforce_solution_consistency(merged, ee_position_orientation),
            ee_position_orientation);
    }

    bool General_7R::has_known_decomposition() const
    {
        return search_joint_index >= 0 && five_r_class != FiveRClass::UNSUPPORTED;
    }

    std::string General_7R::get_kinematic_family() const
    {
        if (!has_known_decomposition())
        {
            return "7R-UNKNOWN";
        }
        return std::string("7R-OFFSET_WRIST[lock j") + std::to_string(locked_joint_index + 1) +
               ", search j" + std::to_string(search_joint_index + 1) + ":" + five_r_family + "]";
    }
}
