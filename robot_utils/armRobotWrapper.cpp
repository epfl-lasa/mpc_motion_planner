#include "armRobotWrapper.hpp"



ArmRobotWrapper::ArmRobotWrapper(std::string urdf_path, const std::string ee_link_name) {

    pinocchio::urdf::buildModel(urdf_path, model);

    pinocchio::Data new_data(model);

    data = new_data;

    //std::cout << "link name in ArmRobotWrapper : " << ee_link_name << std::endl;
    // bool frame_exist;
    // frame_exist = model.existFrame(ee_link_name);
    //std::cout << "frame exist ? " << frame_exist << std::endl;
    frame_id = model.getFrameId(ee_link_name);
    //std::cout << "frame_id in ArmRobotWrapper constructor : " << frame_id << std::endl;
    //std::cout << "frame : " << model.frames[frame_id] << std::endl;
}

Eigen::Matrix<double, 7, 1> ArmRobotWrapper::inverse_kinematic(Eigen::Matrix3d orientation, Eigen::Vector3d position){

    const pinocchio::SE3 oMdes(orientation, position);
    
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    const double eps  = 1e-5; // 5e-5 //1e-4;
    const int IT_MAX  = 1000;
    const double DT   = 1e-1;
    const double damp = 1e-2;
    
    pinocchio::Data::Matrix6x J(6,model.nv);
    J.setZero();
    
    bool success = false;
    Eigen::Matrix<double, 6, 1> err;
    Eigen::VectorXd v(model.nv);
    for (int i=0;;i++)
    {
        pinocchio::forwardKinematics(model,data,q);
        pinocchio::updateFramePlacement(model,data,frame_id);
        const pinocchio::SE3 dMf = oMdes.actInv(data.oMf[frame_id]);
        err = pinocchio::log6(dMf).toVector();

        if(err.norm() < eps)
        {
            success = true;
            break;
        }
        if (i >= IT_MAX)
        {
            success = false;
            break;
        }

        pinocchio::computeFrameJacobian(model, data, q, frame_id, J);
        pinocchio::Data::Matrix6 JJt;
        JJt.noalias() = J * J.transpose();
        JJt.diagonal().array() += damp;
        v.noalias() = - J.transpose() * JJt.ldlt().solve(err);
        q = pinocchio::integrate(model,q,v*DT);
    }

    Eigen::Ref<Eigen::Matrix<double, 7, 1>> qSolution(q);

    return qSolution;

}

Eigen::Matrix<double, NDOF, 1> ArmRobotWrapper::inverse_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity){

    // Construct data
    pinocchio::Data::Matrix6x J(6,7); J.setZero();

    pinocchio::forwardKinematics(model,data,q);
    pinocchio::updateFramePlacement(model,data,frame_id);
    pinocchio::computeFrameJacobian(model, data, q, frame_id, J);

    // For some reasons we should rotate the jacobian
    Eigen::MatrixXd rotateJacobian(6,6); rotateJacobian.setZero();
    rotateJacobian.block(0,0,3,3) = data.oMf[frame_id].rotation();
    rotateJacobian.block(3,3,3,3) = data.oMf[frame_id].rotation();
    J = rotateJacobian * J;

    Eigen::VectorXd joint_velocity(model.nv);
    Eigen::Matrix<double, 6, 1> task_velocity;
    task_velocity << linear_velocity, angular_velocity; 
    
    // Solve using damped pseudo-inverse
    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = J * J.transpose();
    JJt.diagonal().array() += 1e-5;
    joint_velocity.noalias() = J.transpose() * JJt.ldlt().solve(task_velocity);

    // Cast to static matrix
    Eigen::Ref<Eigen::Matrix<double, NDOF, 1>> joint_velocity_solution(joint_velocity);
    return joint_velocity_solution;
}

Eigen::Matrix<double, 3, 4> ArmRobotWrapper::forward_kinematics(Eigen::Matrix<double, NDOF, 1> q, std::string frame_name){
    int frame_id_to_use = model.getFrameId(frame_name);
    pinocchio::forwardKinematics(model,data,q);
    pinocchio::updateFramePlacement(model,data,frame_id_to_use);

    Eigen::MatrixXd x(3, 4); x.setZero();
    x.block(0, 0, 3, 1) = data.oMf[frame_id_to_use].translation();
    x.block(0, 1, 3, 3) = data.oMf[frame_id_to_use].rotation();

    return x;
}

Eigen::Matrix<double, 3, 4> ArmRobotWrapper::forward_kinematics(Eigen::Matrix<double, NDOF, 1> q){
    pinocchio::forwardKinematics(model,data,q);
    pinocchio::updateFramePlacement(model,data,frame_id);

    Eigen::MatrixXd x(3, 4); x.setZero();
    x.block(0, 0, 3, 1) = data.oMf[frame_id].translation();
    x.block(0, 1, 3, 3) = data.oMf[frame_id].rotation();

    return x;
}

Eigen::Matrix<double, 6, 1> ArmRobotWrapper::forward_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Matrix<double, NDOF, 1> qdot){

    pinocchio::Data::Matrix6x J(6,7); J.setZero();

    pinocchio::forwardKinematics(model,data,q);
    pinocchio::updateFramePlacement(model,data,frame_id);
    pinocchio::computeFrameJacobian(model, data, q, frame_id, J);

    // For some reasons we should rotate the jacobian
    Eigen::MatrixXd rotateJacobian(6,6); rotateJacobian.setZero();
    rotateJacobian.block(0,0,3,3) = data.oMf[frame_id].rotation();
    rotateJacobian.block(3,3,3,3) = data.oMf[frame_id].rotation();
    J = rotateJacobian * J;

    Eigen::Matrix<double, 6, 1> task_velocity = J*qdot;

    return task_velocity;
}

Eigen::Matrix<double, 6, 1> ArmRobotWrapper::forward_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Matrix<double, NDOF, 1> qdot, std::string frame_name){

    int frame_id_to_use = model.getFrameId(frame_name);
    pinocchio::Data::Matrix6x J(6,7); J.setZero();

    pinocchio::forwardKinematics(model,data,q);
    pinocchio::updateFramePlacement(model,data,frame_id_to_use);
    pinocchio::computeFrameJacobian(model, data, q, frame_id_to_use, J);

    // For some reasons we should rotate the jacobian
    Eigen::MatrixXd rotateJacobian(6,6); rotateJacobian.setZero();
    rotateJacobian.block(0,0,3,3) = data.oMf[frame_id_to_use].rotation();
    rotateJacobian.block(3,3,3,3) = data.oMf[frame_id_to_use].rotation();
    J = rotateJacobian * J;

    Eigen::Matrix<double, 6, 1> task_velocity = J*qdot;

    return task_velocity;
}

/*
template <typename robotModel>
ArmRobotWrapper<robotModel>::ArmRobotWrapper(std::string urdf_path) {

    pinocchio::urdf::buildModel(urdf_path, model);

    pinocchio::Data new_data(model);

    data = new_data;

    frame_id = model.getFrameId(ee_link_name);
}

template <typename robotModel>
Eigen::Matrix<double, 7, 1> ArmRobotWrapper<robotModel>::inverse_kinematic(Eigen::Matrix3d orientation, Eigen::Vector3d position){

    const pinocchio::SE3 oMdes(orientation, position);
    
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    const double eps  = 1e-4;
    const int IT_MAX  = 1000;
    const double DT   = 1e-1;
    const double damp = 1e-2;
    
    pinocchio::Data::Matrix6x J(6,model.nv);
    J.setZero();
    
    bool success = false;
    Eigen::Matrix<double, 6, 1> err;
    Eigen::VectorXd v(model.nv);
    for (int i=0;;i++)
    {
        pinocchio::forwardKinematics(model,data,q);
        pinocchio::updateFramePlacement(model,data,frame_id);
        const pinocchio::SE3 dMf = oMdes.actInv(data.oMf[frame_id]);
        err = pinocchio::log6(dMf).toVector();
        if(err.norm() < eps)
        {
            success = true;
            break;
        }
        if (i >= IT_MAX)
        {
            success = false;
            break;
        }
        pinocchio::computeFrameJacobian(model, data, q, frame_id, J);
        pinocchio::Data::Matrix6 JJt;
        JJt.noalias() = J * J.transpose();
        JJt.diagonal().array() += damp;
        v.noalias() = - J.transpose() * JJt.ldlt().solve(err);
        q = pinocchio::integrate(model,q,v*DT);
    }

    Eigen::Ref<Eigen::Matrix<double, 7, 1>> qSolution(q);

    return qSolution;

}

template <typename robotModel>
Eigen::Matrix<double, NDOF, 1> ArmRobotWrapper<robotModel>::inverse_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity){

    // Construct data
    pinocchio::Data::Matrix6x J(6,7); J.setZero();

    pinocchio::forwardKinematics(model,data,q);
    pinocchio::updateFramePlacement(model,data,frame_id);
    pinocchio::computeFrameJacobian(model, data, q, frame_id, J);

    // For some reasons we should rotate the jacobian
    Eigen::MatrixXd rotateJacobian(6,6); rotateJacobian.setZero();
    rotateJacobian.block(0,0,3,3) = data.oMf[frame_id].rotation();
    rotateJacobian.block(3,3,3,3) = data.oMf[frame_id].rotation();
    J = rotateJacobian * J;

    Eigen::VectorXd joint_velocity(model.nv);
    Eigen::Matrix<double, 6, 1> task_velocity;
    task_velocity << linear_velocity, angular_velocity; 
    
    // Solve using damped pseudo-inverse
    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = J * J.transpose();
    JJt.diagonal().array() += 1e-5;
    joint_velocity.noalias() = J.transpose() * JJt.ldlt().solve(task_velocity);

    // Cast to static matrix
    Eigen::Ref<Eigen::Matrix<double, NDOF, 1>> joint_velocity_solution(joint_velocity);
    return joint_velocity_solution;
}

template <typename robotModel>
Eigen::Matrix<double, 6, 1> ArmRobotWrapper<robotModel>::forward_velocities(Eigen::Matrix<double, NDOF, 1> q, Eigen::Matrix<double, NDOF, 1> qdot){

    pinocchio::Data::Matrix6x J(6,7); J.setZero();

    pinocchio::forwardKinematics(model,data,q);
    pinocchio::updateFramePlacement(model,data,frame_id);
    pinocchio::computeFrameJacobian(model, data, q, frame_id, J);

    // For some reasons we should rotate the jacobian
    Eigen::MatrixXd rotateJacobian(6,6); rotateJacobian.setZero();
    rotateJacobian.block(0,0,3,3) = data.oMf[frame_id].rotation();
    rotateJacobian.block(3,3,3,3) = data.oMf[frame_id].rotation();
    J = rotateJacobian * J;

    Eigen::Matrix<double, 6, 1> task_velocity = J*qdot;

    return task_velocity;
}

template class ArmRobotWrapper<PandaWrapper>;
template class ArmRobotWrapper<Kuka7Wrapper>;
*/

//template class ArmRobotWrapper<>

