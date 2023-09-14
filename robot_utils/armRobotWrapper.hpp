#pragma once

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/urdf.hpp"

#define NDOF 7

class ArmRobotWrapper {  
    
  public:
    
    pinocchio::Model model;
    pinocchio::Data data;
    int frame_id;

    ArmRobotWrapper(std::string urdf_path, std::string ee_link_name);
    Eigen::Matrix<double, NDOF, 1> inverse_kinematic(Eigen::Matrix3d orientation, Eigen::Vector3d position);
    Eigen::Matrix<double, NDOF, 1> inverse_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity);
    Eigen::Matrix<double, 6, 1> forward_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Matrix<double, NDOF, 1> qdot);
    
};