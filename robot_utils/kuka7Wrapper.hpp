#include "armRobotWrapper.hpp"

#define KUKA7_EE_LINK "grabber_link"

class Kuka7Wrapper: public ArmRobotWrapper {

    //void Kuka7Wrapper::Kuka7Wrapper(std::string urdf, std::string )

    public:
        Kuka7Wrapper(std::string urdf) : ArmRobotWrapper(urdf, KUKA7_EE_LINK) {}

        // Limits from https://frankaemika.github.io/docs/control_parameters.html
        Eigen::Matrix<double, NDOF, 1> min_position {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
        Eigen::Matrix<double, NDOF, 1> max_position {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
        Eigen::Matrix<double, NDOF, 1> max_velocity {2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100};
        Eigen::Matrix<double, NDOF, 1> max_acceleration {15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0};
        Eigen::Matrix<double, NDOF, 1> max_jerk {7500, 3750, 5000, 6250, 7500, 10000, 10000};
        Eigen::Matrix<double, NDOF, 1> max_torque {87, 87, 87, 87, 12, 12, 12};
        double max_torqueDot {1000};

        double max_linear_velocity {1.7};
        double max_angular_velocity {2.5};

        double min_height {0.05};

};