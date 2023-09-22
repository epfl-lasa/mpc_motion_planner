#pragma once

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "armRobotWrapper.hpp"

#define NDOF 7
//#define EE_LINK_NAME "panda_tool"


class Kuka14Wrapper : public ArmRobotWrapper {

  public:
    /*
    pinocchio::Model model;
    pinocchio::Data data;
    int frame_id;

    PandaWrapper(std::string urdf_path);
    Eigen::Matrix<double, NDOF, 1> inverse_kinematic(Eigen::Matrix3d orientation, Eigen::Vector3d position);
    Eigen::Matrix<double, NDOF, 1> inverse_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity);
    Eigen::Matrix<double, 6, 1> forward_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Matrix<double, NDOF, 1> qdot);
    */
    std::string ee_link_name = "grabber_link";
    std::string urdf;
    Kuka14Wrapper(std::string urdf) : ArmRobotWrapper(urdf, ee_link_name) {};

    // Limits from urdf
    Eigen::Matrix<double, NDOF, 1> min_position {-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543};
    Eigen::Matrix<double, NDOF, 1> max_position {2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543};
    Eigen::Matrix<double, NDOF, 1> max_velocity {1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562};
    Eigen::Matrix<double, NDOF, 1> max_acceleration {15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0};
    Eigen::Matrix<double, NDOF, 1> max_jerk {7500, 3750, 5000, 6250, 7500, 10000, 10000};
    Eigen::Matrix<double, NDOF, 1> max_torque {320, 320, 176, 176, 110, 40, 40};
    double max_torqueDot {1000};

    double max_linear_velocity {1.7};
    double max_angular_velocity {2.5};

    double min_height {0.05};
    
};