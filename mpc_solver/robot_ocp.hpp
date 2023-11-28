#ifndef SRC_ROCKET_MPC_HPP
#define SRC_ROCKET_MPC_HPP

#include <math.h>

#include <Eigen/Dense>

#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "polynomials/splines.hpp"

#include "solvers/sqp_base.hpp"
#include "solvers/osqp_interface.hpp"
#include "control/mpc_wrapper.hpp"

#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
// #include "robotDynamic.hpp"
// #include "pinocchio/algorithm/joint-configuration.hpp"

#include <iostream>

// Global variable with rocket parameters and useful methods

using namespace std;

#define POLY_ORDER 3
#define NUM_SEG    6

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ minTime_ocp, /*NX*/ 14, /*NU*/ 7, /*NP*/ 1, /*ND*/ 0, /*NG*/ 8, /*TYPE*/ double)
//POLYMPC_FORWARD_DECLARATION(/*Name*/ minTime_ocp, /*NX*/ 14, /*NU*/ 7, /*NP*/ 1, /*ND*/ 0, /*NG*/ 9, /*TYPE*/ double)

/*
NX : Number of states
NU : Number of inputs
NP : Number of variable parameters
ND : Number of static parameters
NG : Number of constraints
TYPE : scalar type
*/

using namespace Eigen;

// This is the Problem definition !!! 
class minTime_ocp : public ContinuousOCP<minTime_ocp, Approximation, SPARSE>{
public:

    ~minTime_ocp() = default;
    int frame_id;

    pinocchio::Model model;

    void init(std::string urdf_path, std::string ee_link_name){
        /*
        std::cout << ee_link_name << std::endl;
        std::cout << "urdf : " << urdf_path << std::endl;
        std::cout << "ee_link_name : " << ee_link_name << std::endl;
        */


       pinocchio::urdf::buildModel(urdf_path, model);
        //std::cout << "getFrameId before" << endl;
        //frame_id = model.getFrameId("panda_tool");
        
        
        frame_id = model.getFrameId(ee_link_name);
        //frame_id = model.getFrameId("panda_tool");
        //std::cout << "getFrameId after" << frame_id << std::endl;
    }

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {

        // -------------- Differential equation ---------------------
        // State vector is x = [pos, vel] in R^14
        // Position variation is joint velocity
        //std::cout << "dynamics_impl" << std::endl;

        xdot.head(7) = x.segment(7, 7); 

        // Joint velocity variation is input = acceleration
        xdot.segment(7, 7) = u; 

        // Scaling dynamic with time parameter        
        xdot *= p(0);

        // std::cout << "p(0) in OCP : " << p(0) << std::endl;
        
        polympc::ignore_unused_var(t);

        //std::cout << "dynamics_impl fin" << std::endl;
    }






EIGEN_STRONG_INLINE constraint_t<scalar_t> evalConstraints(const Ref<const state_t<scalar_t>> x, const Ref<const control_t<scalar_t>> u) const noexcept
{
    //std::cout << "constraint" << std::endl;
    Eigen::Matrix<double, 7, 1> q = x.head(7);
    Eigen::Matrix<double, 7, 1> q_dot = x.tail(7);
    Eigen::Matrix<double, 7, 1> q_dot_dot = u;

    pinocchio::Data data(model);
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacement(model,data,frame_id);

    constraint_t<scalar_t> ineq_constraint;
    ineq_constraint << pinocchio::rnea(model, data, q, q_dot, q_dot_dot),  data.oMf[frame_id].translation()[2];

    // std::cout << "----\n" << ineq_constraint.transpose() << "\n----\n";
    //std::cout << "constraint fin" << std::endl;

    return ineq_constraint;
}
/*
dynamics_impl
dynamics_impl fin
inequality_constraints_impl
constraint ad scalar_t
*/
EIGEN_STRONG_INLINE constraint_t<ad_scalar_t> evalConstraints(const Ref<const state_t<ad_scalar_t>> x, const Ref<const control_t<ad_scalar_t>> u) const noexcept
{
    //std::cout << "constraint ad scalar_t" << std::endl;
    Eigen::Matrix<double, 7, 1> q;
    Eigen::Matrix<double, 7, 1> q_dot;
    Eigen::Matrix<double, 7, 1> q_dot_dot;

    for(int i = 0; i<7; i++){
        q(i) = x(i).value();
        q_dot(i) = x(i+7).value();
        q_dot_dot(i) = u(i).value();
    }
    
    pinocchio::Data data(model);


    // Allocate result container
    Eigen::MatrixXd djoint_torque_dq = Eigen::MatrixXd::Zero(model.nv,model.nv);
    Eigen::MatrixXd djoint_torque_dv = Eigen::MatrixXd::Zero(model.nv,model.nv);
    Eigen::MatrixXd djoint_torque_da = Eigen::MatrixXd::Zero(model.nv,model.nv);

    // Recursive Newton Euler Algorithm to compute derivatives
    pinocchio::computeRNEADerivatives(model, data, q, q_dot, q_dot_dot, djoint_torque_dq, djoint_torque_dv, djoint_torque_da);

    // Compute inverse dynamic : Torque to reach q, q_dot, q_ddot
    pinocchio::rnea(model, data, q, q_dot, q_dot_dot);

    // Composite Rigid Body Algorithm : Computes the upper triangular part of the joint space inertia matrix
    pinocchio::crba(model, data, q);
    data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();

    Eigen::MatrixXd djoint_torque_dtime_f = djoint_torque_dv*q_dot + djoint_torque_da*q_dot_dot;


    constraint_t<ad_scalar_t> ineq_constraint; // ad means automatic differentiation, using ad_scalar_t = Eigen::AutoDiffScalar<derivatives_t>;


    // Fill in torque derivatives
    Eigen::Matrix<scalar_t, 1, NX + NU + NP> jac_row;
    jac_row.setZero();
    for(int i = 0; i < NG-1; i++)                                                                           // for(int i = 0; i < NG-1; i++)
    {   
        // Overwitting with PINOCCHIO data
        jac_row.head(7) = djoint_torque_dq.row(i); // Replace the first 7 elements
        jac_row.segment(7, 7) = djoint_torque_dv.row(i); // Replace 7 elements, starting at pos 7
        jac_row.segment(14, 7) = data.M.row(i); // Replace 7 elements, starting from pos 14

        jac_row(21) = djoint_torque_dtime_f(i);

        ineq_constraint(i).value() = data.tau(i);
        ineq_constraint(i).derivatives() = jac_row;
    }

    // Fill in height derivative
    pinocchio::Data::Matrix6x J(6,7); J.setZero();

    pinocchio::forwardKinematics(model,data,q);

    pinocchio::updateFramePlacement(model,data,frame_id);

    pinocchio::computeFrameJacobian(model, data, q, frame_id, J);

    // For some reasons we should rotate the jacobian
    Eigen::MatrixXd rotateJacobian(6,6); rotateJacobian.setZero();
    rotateJacobian.block(0,0,3,3) = data.oMf[frame_id].rotation();
    rotateJacobian.block(3,3,3,3) = data.oMf[frame_id].rotation();
    J = rotateJacobian * J;

    ineq_constraint(NG-1).value() = data.oMf[frame_id].translation()[2];                    // ineq_constraint(NG-1).value() = data.oMf[frame_id].translation()[2];
    jac_row = Eigen::Matrix<scalar_t, 1, NX + NU + NP>::Zero();
    jac_row.head(7) = J.row(2);
    ineq_constraint(NG-1).derivatives() = jac_row;                                          // ineq_constraint(NG-1).derivatives() = jac_row;


    // // Velocity constraints on the end effector
    // pinocchio::Data::Matrix6x J_partial(6, 7); J_partial.setZero();
    // int frame_last_link_id = model.getFrameId("panda_link7");
    // pinocchio::updateFramePlacement(model,data,frame_last_link_id);
    // pinocchio::computeFrameJacobian(model, data, q, frame_last_link_id, J_partial);

    // //std::cout << "J_partial : " << J_partial << std::endl;
    // //std::cout << "q_dot.head(6) : " << q_dot.head(6) << std::endl;

    // Eigen::MatrixXd v = J_partial * q_dot;

    // //std::cout << "v.block(0, 0, 3, 1) : " << v.block(0, 0, 3, 1) << std::endl;
    // // std::cout << "v.block(0, 0, 3, 1).squaredNorm() :" << v.block(0, 0, 3, 1).squaredNorm() << std::endl;

    // ineq_constraint(NG-2).value() = v.block(0, 0, 3, 1).squaredNorm();

    // //std::cout << "Finished" << std::endl;

    return ineq_constraint;
}

EIGEN_STRONG_INLINE constraint_t<ad2_scalar_t> evalConstraints(const Ref<const state_t<ad2_scalar_t>> x, const Ref<const control_t<ad2_scalar_t>> u) const noexcept
{
    return constraint_t<ad2_scalar_t>::Zero();
}


template<typename T>
EIGEN_STRONG_INLINE void
inequality_constraints_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                            const Ref<const parameter_t <T>> p, const Ref<const static_parameter_t> d,
                            const scalar_t &t, Ref<constraint_t < T>>g) const noexcept
{
    //std::cout << "inequality_constraints_impl" << std::endl;
    g = evalConstraints(x, u);
    //std::cout << "inequality_constraints_impl fin" << std::endl;
}
    


    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                   const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        //std::cout << "lagrange_term_impl" << std::endl;
        lagrange = (T)0.0;
        //std::cout << "lagrange_term_impl fin" << std::endl;
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {   
        //std::cout << "mayer_term_impl" << std::endl;
        mayer = p(0);

        //std::cout << "Mayer's term : " << mayer << std::endl; 

        // polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        //std::cout << "mayer_term_impl fin" << std::endl;
  
    }
};

#endif //SRC_ROCKET_MPC_HPP