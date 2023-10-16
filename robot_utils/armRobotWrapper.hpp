#pragma once

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/urdf.hpp"

#define NDOF 7


//template <typename robotModel>
//class ArmRobotWrapper : public robotModel {  
class ArmRobotWrapper {
    
  public:
    
    //std::string robotModel;
    pinocchio::Model model;
    pinocchio::Data data;
    int frame_id;
    std::string ee_link_name; // Maybe it overwrites 

    //virtual std::string get_ee_link_name() const noexcept;
    ArmRobotWrapper(std::string urdf_path, std::string ee_link_name);
    Eigen::Matrix<double, NDOF, 1> inverse_kinematic(Eigen::Matrix3d orientation, Eigen::Vector3d position);
    Eigen::Matrix<double, NDOF, 1> inverse_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity);
    Eigen::Matrix<double, 6, 1> forward_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Matrix<double, NDOF, 1> qdot);
    Eigen::Matrix<double, 3, 1> forward_kinematics(Eigen::Matrix<double, NDOF, 1> q);

    /*Eigen::Matrix<double, NDOF, 1> min_position;
    Eigen::Matrix<double, NDOF, 1> max_position;
    Eigen::Matrix<double, NDOF, 1> max_velocity;
    Eigen::Matrix<double, NDOF, 1> max_acceleration;
    Eigen::Matrix<double, NDOF, 1> max_jerk;
    Eigen::Matrix<double, NDOF, 1> max_torque;
    double max_torqueDot;

    double max_linear_velocity;
    double max_angular_velocity;

    double min_height;*/
    
};