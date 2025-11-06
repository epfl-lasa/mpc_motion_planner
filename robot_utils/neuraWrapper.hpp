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



class NeuraWrapper : public ArmRobotWrapper {

  public:
    /*
    pinocchio::Model model;
    pinocchio::Data data;
    int frame_id;

    NeuraWrapper(std::string urdf_path);
    Eigen::Matrix<double, NDOF, 1> inverse_kinematic(Eigen::Matrix3d orientation, Eigen::Vector3d position);
    Eigen::Matrix<double, NDOF, 1> inverse_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity);
    Eigen::Matrix<double, 6, 1> forward_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Matrix<double, NDOF, 1> qdot);
    */

    std::string ee_link_name = "maira7M_grasptarget";
    std::string urdf;
    NeuraWrapper(std::string urdf) : ArmRobotWrapper(urdf, "maira7M_grasptarget") {/*std::cout << "Using NeuraWrapper" << std::endl;*/};

    //std::string get_ee_link_name(){ return "panda_tool"; } override

    // Limits from TODO 
    Eigen::Matrix<double, NDOF, 1> min_position {-3.14159265, -2.0943951 , -3.14159265, -2.61799388, -3.14159265, -2.53072742, -3.14159265};
    Eigen::Matrix<double, NDOF, 1> max_position {3.14159265, 2.0943951 , 3.14159265, 2.61799388, 3.14159265, 2.53072742, 3.14159265};
    Eigen::Matrix<double, NDOF, 1> max_velocity {2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.0943951};
    Eigen::Matrix<double, NDOF, 1> max_acceleration {15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0}; // TODO 
    Eigen::Matrix<double, NDOF, 1> max_jerk {7500, 3750, 5000, 6250, 7500, 10000, 10000}; // TODO 
    Eigen::Matrix<double, NDOF, 1> max_torque {87, 87, 87, 87, 12, 12, 12}; // TODO 
    double max_torqueDot {1000};

    double max_linear_velocity {4.01}; // According to GUI is 4.5 but in robot limits:4.01
    double max_angular_velocity {3.14}; // same as Panda; in GUI 6.28 rad/s

    double min_height {0.4}; // TODO
    
};

