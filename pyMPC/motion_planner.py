import mpc_solver.motion_planning_lib as mpl
import numpy as np
import enum, time
import os, sys
from pathlib import Path

""" CONVENTION :
Trajectories dimensions are always in this order : (Ntraj, Npts, NDOF)
Where Ntraj is the number of different trajectories, Npts is the number of time-steps and NDOF the number of degrees of freedom
"""

sys.path.append(os.path.join(os.getcwd(), '../'))
import descriptions.robot_descriptions.franka_panda_bullet.franka_panda as panda_utils
import descriptions.robot_descriptions.Kuka_iiwa7_and_14_models.kuka_iiwa_7 as kuka7_utils
import descriptions.robot_descriptions.Kuka_iiwa7_and_14_models.kuka_iiwa_14 as kuka14_utils

class RobotModel(enum.Enum):
    Panda = 1
    Kuka7 = 2
    Kuka14 = 3

CONS_MARGINS = [0.9, 0.9, 0.4, 0.7, 0.1]
LINE_SEARCH_MAX_ITER = 10
SQP_MAX_ITER = 2

#----------------------------------------------------------------------------------------------#

class Trajectory():
    def __init__(self, utils, input_traj=None, ruckig=False):
        self._utils = utils
        if input_traj is not None:
            self._t, self._q, self._qdot, self._qddot, self._tau = reshape_traj_from_tupple_to_numpy(input_traj)
        else:
            self._t, self._q, self._qdot, self._qddot, self._tau = None, None, None, None, None
        self._ruckig = ruckig

    def _state_cons_satisfied(self, state:np.ndarray, limits:np.ndarray, verbose:bool=False) -> np.ndarray:
        """
        Check constraints satisfaction for a given state and its limits.
        _______________________________________________________________________________________________________________
        Input :
            state   (Ntraj x NPTS x NDOF)   :   np.ndarray containing the actual state values
            limits  (NDOF, 2)               :   np.ndarray of dim  representing the state min and max acceptable values
        Return :
            state_cons_satisfied    (Ntraj) :   np.ndarray (bool) containing a boolean that indicates
                                                wheter or not constraints are satisfied.
        _______________________________________________________________________________________________________________
        """
        state_cons_satisfied = np.ndarray(shape=(state.shape[0]))
        which_state_not_satisfied = np.ndarray(shape=(state.shape[0], state.shape[-1]))
        which_joint_not_satisfied = []
        for i, traj_i_state in enumerate(state):
            traj_i_state_cons_not_satisfied = np.logical_or(traj_i_state > limits[:, 1], traj_i_state < limits[:, 0])
            which_state_not_satisfied[i] = np.sum(traj_i_state_cons_not_satisfied, axis=0) > np.zeros(shape=(state.shape[-1],)) # Detect which state does not satisfy cons
            traj_i_state_cons_not_satisfied = np.sum(traj_i_state_cons_not_satisfied, axis=-1)
            state_cons_satisfied[i] = np.sum(traj_i_state_cons_not_satisfied) == 0

        if verbose:
            return state_cons_satisfied, which_state_not_satisfied
    
        return state_cons_satisfied

    def _q_cons_satisfied(self, verbose:bool=False) -> np.ndarray:
        """
        Check constraints satisfaction for joint positions.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint position constraints are satisfied.
        _______________________________________________________________________________________________________________
        """
        return self._state_cons_satisfied(self._q, self._utils.X_limits, verbose=verbose)

    def _qdot_cons_satisfied(self, verbose:bool=False) -> np.ndarray:
        """
        Check constraints satisfaction for joint velocities.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint velocity constraints are satisfied.
        _______________________________________________________________________________________________________________
        """
        return self._state_cons_satisfied(self._qdot, self._utils.V_limits, verbose=verbose)

    def _qddot_cons_satisfied(self, verbose:bool=False) -> np.ndarray:
        """
        Check constraints satisfaction for joint accelerations.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint acceleration constraints are satisfied.
        _______________________________________________________________________________________________________________
        """
        return self._state_cons_satisfied(self._qddot, self._utils.A_limits, verbose=verbose)

    def _tau_cons_satisfied(self, verbose:bool=False) -> np.ndarray:
        """
        Check constraints satisfaction for joint torques.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint torque constraints are satisfied.
        _______________________________________________________________________________________________________________
        """
        T_limits = np.ndarray(shape=(self._tau.shape[-1], 2))
        T_limits[:, 0] = -self._utils.T_limits
        T_limits[:, 1] = self._utils.T_limits
        return self._state_cons_satisfied(self._tau, T_limits, verbose=verbose)

    def _power(self, verbose:bool=False) -> np.ndarray:
        power = np.ndarray(shape=self._t.shape)
        for i, (qdot, tau) in enumerate(zip(np.abs(self._qdot), np.abs(self._tau))):
            power[i] = np.sum(qdot * tau, axis=-1)
        
        return power
    
    def _get_numerical_integral_of(self, state_str, init_state):
        """TODO"""
        integral = np.zeros_like(self._q)
        state_to_integrate = self.__getitem__(state_str)
        for i, (t_i, traj_state_i) in enumerate(zip(self._t, state_to_integrate)):
            integral[i, 0] = init_state[i]
            for k in range(t_i.shape[0] - 1):
                integral[i, k+1] = integral[i, k] + (traj_state_i[k]+traj_state_i[k+1]) * (t_i[k+1]-t_i[k]) / 2

        return integral
    
    def _get_numerical_derivative_of(self, state_str):
        """TODO"""
        derivative = np.zeros_like(self._q)
        state_to_derive = self.__getitem__(state_str)
        for i, (t_i, traj_state_i) in enumerate(zip(self._t, state_to_derive)):
            derivative[i, 0] = 0
            for k in range(t_i.shape[0] - 1):
                derivative[i, k+1] = (traj_state_i[k+1]-traj_state_i[k]) / (t_i[k+1]-t_i[k])

        return derivative

    def _set_t(self, t:np.ndarray) -> None:
        """
        Set value of the time vector.
        _______________________________________________________________________________________________________________
        Input :
            t   (Ntraj, NPTS)  :    np.ndarray (float) containing the time steps for each trajectory
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        self._t = t

    def _set_q(self, q:np.ndarray) -> None:
        """
        Set value of the joint position vector.
        _______________________________________________________________________________________________________________
        Input :
            q   (Ntraj, NPTS, NDOF) :   np.ndarray (float) containing the joint positions for each trajectory
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        self._q = q

    def _set_qdot(self, qdot:np.ndarray) -> None:
        """
        Set value of the joint velocity vector.
        _______________________________________________________________________________________________________________
        Input :
            qdot    (Ntraj, NPTS, NDOF) :   np.ndarray (float) containing the joint velocities for each trajectory
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        self._qdot = qdot

    def _set_qddot(self, qddot:np.ndarray) -> None:
        """
        Set value of the joint acceleration vector.
        _______________________________________________________________________________________________________________
        Input :
            qddot   (Ntraj, NPTS, NDOF) :   np.ndarray (float) containing the joint accelerations for each trajectory
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        self._qddot = qddot

    def _set_tau(self, tau:np.ndarray) -> None:
        """
        Set value of the joint torque vector.
        _______________________________________________________________________________________________________________
        Input :
            tau (Ntraj, NPTS, NDOF) :   np.ndarray (float) containing the joint torques for each trajectory
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        self._tau = tau

    def __getitem__(self, slice):
        """
        Access items of the object using brackets []
        _______________________________________________________________________________________________________________
        Input :
            slice (1)   : str or int, that specifies which value to access
        Return :
            -           : np.ndarray (float), the desired value specified by <slice>  
        _______________________________________________________________________________________________________________
        """
        if isinstance(slice, str):
            if slice=="t":
                return self._t
            elif slice=="q":
                return self._q
            elif slice=="qdot":
                return self._qdot
            elif slice=="qddot":
                return self._qddot
            elif slice=="tau":
                return self._tau
            elif slice=="power":
                return self._power()
        elif isinstance(slice, int):
            return self._t[slice], self._q[slice], self._qdot[slice], self._qddot[slice], self._tau[slice]
        else:
            raise ValueError("######## Trajectory Error : Index must be an integer or a string but is now a {} ########" % (type(slice)))
            
    def __add__(self, other):
        """
        Defines "+" operator as a concatenation. For two given Trajectory object traj1, traj2, the operation
        traj1 + traj2 returns a new trajectory object that contains both traj1 and traj2 data.
        For instance, if traj1.shape[0] = N1 and traj2.shape[0] = N2, then traj.shape[0] = N1 + N2, as the "+"
        operator act as a concatenation in this context.
        _______________________________________________________________________________________________________________
        Input :
            other : Trajectory object
        Return :
            result : Trajectory object that contains both <self> and <other>  
        _______________________________________________________________________________________________________________
        """
        assert isinstance(other, Trajectory)
        if self._t is not None:
            assert(self._t.shape[1] == other._t.shape[1])
            t = np.concatenate((self._t, other.t), axis=0)
            q = np.concatenate((self._q, other.q), axis=0)
            qdot = np.concatenate((self._qdot, other.qdot), axis=0)
            qddot = np.concatenate((self._qddot, other.qddot), axis=0)
            tau = np.concatenate((self._tau, other.tau), axis=0)

        else:
            t, q, qdot, qddot, tau  = other.t, other.q, other.qdot, other.qddot, other.tau

        result = Trajectory(self._utils, ruckig=self._ruckig)
        result._set_t(t)
        result._set_q(q)
        result._set_qdot(qdot)
        result._set_qddot(qddot)
        result._set_tau(tau)

        return result

    def __len__(self):
        """
        Defines the object length as the number of concatenated trajectories stored within it.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            len (1) :   Number of concatenated trajectories
        _______________________________________________________________________________________________________________
        """
        return self._t.shape[0]

    def __eq__(self, other):
        """Overload == function""" 
        raise NotImplementedError("######## Trajectory Error : Unvalid operation == for Trajectory class ########")

    def __ne__(self, other):
        """Overload != function"""
        raise NotImplementedError("######## Trajectory Error : Unvalid operation != for Trajectory class ########")

    @property
    def all_cons_satisfied(self) -> np.ndarray:
        """
        Check that q, qdot, qddot and tau constraints are satisfied. Implemented as a property, therefore not callable.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            all_cons_satisfied (Ntraj) :    np.ndarray (bool) containing True if all the constraints are satisfied, and
                                            False otherwise, for each trajectory.
        """
        q_cons = self._q_cons_satisfied()
        qdot_cons = self._qdot_cons_satisfied()
        qddot_cons = self._qddot_cons_satisfied()
        tau_cons = self._tau_cons_satisfied()
        
        return np.logical_and(np.logical_and(q_cons,qdot_cons), np.logical_and(qddot_cons, tau_cons))
        
    @property
    def q_cons_satisfied(self):
        return self._q_cons_satisfied(verbose=True)

    @property
    def qdot_cons_satisfied(self):
        return self._qdot_cons_satisfied(verbose=True)

    @property
    def qddot_cons_satisfied(self):
        return self._qddot_cons_satisfied(verbose=True)
    
    @property
    def tau_cons_satisfied(self):
        return self._tau_cons_satisfied(verbose=True)
    
    @property
    def t(self):
        return self._t
    
    @property
    def q(self):
        return self._q
    
    @property
    def qdot(self):
        return self._qdot
    
    @property
    def qddot(self):
        return self._qddot
    
    @property
    def tau(self):
        return self._tau
    
    @property
    def power(self):
        return self._power()
    
    @property
    def shape(self):
        return self._t.shape
    
    @property
    def duration(self):
        duration = np.ndarray(shape=(self._t.shape[0],))
        for i, t in enumerate(self._t):
            duration[i] = t[-1] - t[0]
        return duration

    @property
    def type(self):
        if self._ruckig:
            return "Ruckig"
        else:
            return "Polympc"

#----------------------------------------------------------------------------------------------#

class MotionPlanner():
    def __init__(self, robot_model):
        """
        Instantiate a MotionPlanner object that will be used to solve the motion planning problem.
        _______________________________________________________________________________________________________________
        Input :
            robot_model    (1)  :   RobotModel (enum) that indicates which robot to use
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        parent_dir = Path(__file__).parent.parent
        if robot_model == RobotModel.Panda:
            self._robot_utils = panda_utils
            self._motion_planner = mpl.PandaMotionPlanner(os.path.join(parent_dir, self._robot_utils.MPC_ROBOT_URDF_PATH))
        elif robot_model == RobotModel.Kuka7:
            self._robot_utils = kuka7_utils
            self._motion_planner = mpl.Kuka7MotionPlanner(os.path.join(parent_dir, self._robot_utils.MPC_ROBOT_URDF_PATH))
        elif robot_model == RobotModel.Kuka14:
            self._robot_utils = kuka14_utils
            self._motion_planner = mpl.Kuka14MotionPlanner(os.path.join(parent_dir, self._robot_utils.MPC_ROBOT_URDF_PATH))

        self._robotModel = robot_model
        self._x0, self._xd = None, None
        self._info, self._cons_margins = None, CONS_MARGINS
        self._motion_planner.set_constraint_margins(*self._cons_margins)

    def set_acceleration_constraints(self, max_acceleration:np.ndarray) -> None:
        """
        Set the acceleration constraints for the motion planner.
        _______________________________________________________________________________________________________________
        Input :
            max_acceleration    (NDOF)  :   np.ndarray (float) containing the maximum acceleration for each joint
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        max_acceleration = max_acceleration.squeeze() # Remove useless dimensions
        assert(len(max_acceleration.shape) == 1)
        assert(max_acceleration.shape[0] == self._robot_utils.NDOF)
        assert(np.all(max_acceleration > 0))
        self._motion_planner.set_acceleration_constraints(max_acceleration)

    def set_target_state(self, xd:tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Set the target state xd = (q, dq/dt, d²q/dt²) for the motion planner to reach.
        _______________________________________________________________________________________________________________
        Input :
            xd  (3) :   tuple of np.ndarray that contains the desired joint positions, velocities and accelerations.
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        assert(len(xd) == 3)
        for xd_i in xd:
            xd_i = xd_i.squeeze() # Remove useless dimensions
            assert(len(xd_i.shape) == 1)
            assert(xd_i.shape[0] == self._robot_utils.NDOF)
        
        self._xd = xd
        self._motion_planner.set_target_state(*xd)

    def set_current_state(self, x0:tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Set the current state x0 = (q, dq/dt, d²q/dt²) for the motion planner to start from.
        _______________________________________________________________________________________________________________
        Input :
            x0  (3) :   tuple of np.ndarary that contains the current joint positions, velocities and accelerations.
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        assert len(x0) == 3, "x0 should be a 3D-tuple!"
        for x0_i in x0:
            assert len(x0_i.shape) == 1, "q, qdot, qddot should be a 1D-numpy array!"
            assert x0_i.shape[0] == self._robot_utils.NDOF, "Length of q, qdot, qddot must match the number of DOFs!"
        
        self._x0 = x0
        self._motion_planner.set_current_state(*x0)

    def get_trajectory(self, ruckig:bool=False) -> Trajectory:
        """
        Get either the trajectory solved using polympc or the trajectory solved with ruckig (if ruckig=True). A
        trajectory must be available and therefore you should have called the .solve() method at least once before.
        _______________________________________________________________________________________________________________
        Input :
            *ruckig  (1) :  bool to choose between the polympc trajectory (ruckig=False) or the ruckig one (ruckig=True)
        Return :
            traj    : Trajectory object that contains the desired trajectory
        _______________________________________________________________________________________________________________
        """
        if self._info is None:
            raise RuntimeError("You must have called the .solve() method before calling .get_trajectory()")

        if ruckig:
            traj = Trajectory(self._robot_utils, input_traj=self._motion_planner.get_ruckig_trajectory(), ruckig=True)
        else:     
            # If ruckig was solved before, this will return the ruckig trajectory
            # because at the end of warm_start_RK, the ruckig trajectory is stored
            # in the MPC object using mpc.x_guess(x_guess), mpc.u_guess(u_guess), etc..   
            traj = Trajectory(self._robot_utils, input_traj=self._motion_planner.get_MPC_trajectory(), ruckig=False)
        
        return traj

    def solve(self, ruckig_as_warm_start:bool=True, ruckig:bool=False, sqp_max_iter:int=SQP_MAX_ITER, line_search_max_iter:int=LINE_SEARCH_MAX_ITER) -> dict:
        """
        With default arguments, solves the polympc problem using ruckig as a warm start. If ruckig=True, the generated
        trajectory does only come from ruckig.
        _______________________________________________________________________________________________________________
        Input :
            *ruckig_as_warm_start (1)   :   bool to choose wheter or not ruckig is used as a warm start for polympc
            *ruckig (1)                 :   bool to choose between the polympc trajectory (ruckig=False) or the ruckig one (ruckig=True)
            *sqp_max_iter (1)           :   int to choose the max number of SQP iterations to use in polympc
            *line_search_max_iter (1)   :   int to choose the max number of line search iterations to use in polympc
        Return :
            info    :   dictionnary with fields "status" (1 if polympc converged, 0 otherwise), "iter" (number of sqp
                        iterations to converge) and "time_to_solve".
        _______________________________________________________________________________________________________________
        """
        if self._x0 is not None and self._xd is not None:
            if ruckig:
                start = time.time()
                self._motion_planner.solve_ruckig_trajectory()
                time_to_solve = time.time() - start
            else:
                start = time.time()
                self._motion_planner.solve_trajectory(ruckig_as_warm_start, sqp_max_iter, line_search_max_iter)
                time_to_solve = time.time() - start

            status, iter = self._motion_planner.get_mpc_info()
            info = {"status": status, "iter": iter, "time_to_solve": time_to_solve}
            self._info = info
        else:
            raise ValueError("######## MotionPlanner Error : Initial or/and current state are None ########")
        
        return info
    
    def set_constraints_margins(self, cons_margins:list=CONS_MARGINS) -> None:
        """
        Set the desired constraints margins to use in the solver.
        _______________________________________________________________________________________________________________
        Input :
            *cons_margins (5)  :    list of float between 0 and 1 that defines the margin w.r.t q, qdot, qddot, torque, 
                                    jerk
        Return :
            None
        _______________________________________________________________________________________________________________
        """
        self._cons_margins = cons_margins
        self._motion_planner.set_constraint_margins(*cons_margins)
    
    def forward_kinematics(self, q:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the end-effector position given the joint positions.
        _______________________________________________________________________________________________________________
        Input :
            q (Npts, NDOF) or (NDOF)    :   np.ndarray representing the joint positions
        Return :
            ee_pos (Npts, 3) or (3)         :   np.ndarray that contains the 3D end effector position
            ee_rot (Npts, 3, 3) or (3, 3)   :   np.ndarray that contains the end effector rotation matrix
        _______________________________________________________________________________________________________________    
        """
        assert q.shape[-1] == self._robot_utils.NDOF, "######## MotionPlanner Error : q must have the same number of DOF as the robot ########"
        if len(q.shape) == 1: # If q is a single vector
            return self._single_vector_forward_kinematics(q)
        elif len(q.shape) == 2: # If q is a trajectory
            return self._single_trajectory_forward_kinematics(q)
        elif len(q.shape) == 3: # If q is a list of trajectories
            return self._list_of_trajectories_forward_kinematics(q)
        else:
            raise ValueError("######## MotionPlanner Error : q must be a 1D, 2D or 3D np.ndarray ########")
    
    def _single_vector_forward_kinematics(self, q:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._motion_planner.forward_kinematics(q)

    def _single_trajectory_forward_kinematics(self, q:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = np.ndarray(shape=(q.shape[0], 3)), np.ndarray(shape=(q.shape[0], 3, 3))
        for i, qi in enumerate(q):
            pos[i], rot[i] = self._motion_planner.forward_kinematics(qi)
        return pos, rot

    def _list_of_trajectories_forward_kinematics(self, q:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = np.ndarray(shape=(q.shape[0], q.shape[1], 3)), np.ndarray(shape=(q.shape[0], q.shape[1], 3, 3))
        for i, traj_i in enumerate(q):
            pos[i], rot[i] = self._single_trajectory_forward_kinematics(traj_i)
        return pos, rot

    def inverse_kinematics(self, ee_pos:np.ndarray, ee_rot:np.ndarray=None) -> np.ndarray:
        """
        Compute the joint positions given the end-effector position.
        _______________________________________________________________________________________________________________
        Input :
            ee_pos (Ntraj, Npts, 3) or (Npts, 3) or (3)             :   np.ndarray that contains the 3D end effector position
            ee_rot (Ntraj, Npts, 3, 3) or (Npts, 3, 3) or (3, 3)    :   np.ndarray that contains the end effector rotation matrix 
        Return :
            q (Ntraj, Npts, NDOF) or (Npts, NDOF) or (Npts)         :   np.ndarray representing the joint positions
        _______________________________________________________________________________________________________________
        """
        if len(ee_pos.shape) == 1:
            return self._single_vector_inverse_kinematics(ee_pos, ee_rot)
        elif len(ee_pos.shape) == 2:
            return self._single_trajectory_inverse_kinematics(ee_pos, ee_rot)
        elif len(ee_pos.shape) == 3:
            return self._list_of_trajectories_inverse_kinematics(ee_pos, ee_rot)
        else:
            raise ValueError("######## MotionPlanner Error : ee_pos must be a 1D, 2D or 3D np.ndarray ########")
        
    def _single_vector_inverse_kinematics(self, ee_pos:np.ndarray, ee_rot:np.ndarray=None) -> np.ndarray:
        return self._motion_planner.inverse_kinematics(ee_rot, ee_pos)
    
    def _single_trajectory_inverse_kinematics(self, ee_pos:np.ndarray, ee_rot:np.ndarray=None) -> np.ndarray:
        q = np.ndarray(shape=(ee_pos.shape[0], self._robot_utils.NDOF))
        for i, (ee_pos_i, ee_rot_i) in enumerate(zip(ee_pos, ee_rot)):
            q[i] = self._motion_planner.inverse_kinematics(ee_rot_i, ee_pos_i)
        return q
    
    def _list_of_trajectories_inverse_kinematics(self, ee_pos:np.ndarray, ee_rot:np.ndarray=None) -> np.ndarray:
        q = np.ndarray(shape=(ee_pos.shape[0], ee_pos.shape[1], self._robot_utils.NDOF))
        for i, traj_i in enumerate(ee_pos):
            q[i] = self._single_trajectory_inverse_kinematics(traj_i, ee_rot[i])
        return q 

    def sample_state(self, N:int=1, set_qddot_to_zero:bool=False, use_margins:bool=True, speed_feasible_for_ruckig:bool=False, acceleration_feasible_for_ruckig:bool=False, fraction_to_use=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random state within bounds.
        _______________________________________________________________________________________________________________
        Input :
            *set_qddot_to_zero (bool)                   :   boolean that indicates wheter or not to set the acceleration to zero
            *use_margins (bool)                         :   boolean that indicates wheter or not to consider the margins when sampling
            *speed_feasible_for_ruckig (bool)           :   boolean that indicates wheter or not to consider the speed feasibility
            *acceleration_feasible_for_ruckig (bool)    :   boolean that indicates wheter or not to consider the acceleration feasibility
            *fraction_to_use (bool)                     :   float between 0 and 1 that indicates the fraction of the feasible state space to use
        Return :
            x   (3) :   tuple of np.ndarray that contains the desired joint positions, velocities and accelerations.
        _______________________________________________________________________________________________________________
        """
        assert N>=1, "######## MotionPlanner Error : N must be greater or equal to 1 ########"

        if use_margins:
            safety_range_pos = (1-CONS_MARGINS[0])*(self._robot_utils.X_limits[:, 1] - self._robot_utils.X_limits[:, 0])/2
            xmin = self._robot_utils.X_limits[:, 0] + safety_range_pos
            xmax = self._robot_utils.X_limits[:, 1] - safety_range_pos
            vmin = self._robot_utils.V_limits[:, 0] * CONS_MARGINS[1]
            vmax = self._robot_utils.V_limits[:, 1] * CONS_MARGINS[1]
            amin = self._robot_utils.A_limits[:, 0] * CONS_MARGINS[2]
            amax = self._robot_utils.A_limits[:, 1] * CONS_MARGINS[2]
        else:
            xmin = self._robot_utils.X_limits[:, 0]
            xmax = self._robot_utils.X_limits[:, 1]
            vmin = self._robot_utils.V_limits[:, 0]
            vmax = self._robot_utils.V_limits[:, 1]
            amin = self._robot_utils.A_limits[:, 0]
            amax = self._robot_utils.A_limits[:, 1]

        if speed_feasible_for_ruckig:
            assert False, "Not implemented yet"

        if acceleration_feasible_for_ruckig:
            assert False, "Not implemented yet"

        q = np.random.random((N, self._robot_utils.NDOF))
        q *= (xmax - xmin) * fraction_to_use
        q += xmin

        # Velocity : rand * (ub - lb) + lb
        qdot = np.random.random((N, self._robot_utils.NDOF))
        qdot *= (vmax - vmin) * fraction_to_use
        qdot += vmin

        if set_qddot_to_zero:
            qddot = np.zeros((N, self._robot_utils.NDOF))
        else:
            # Acceleration : rand * (ub - lb) + lb
            qddot = np.random.random((N, self._robot_utils.NDOF))
            qddot *= (amax - amin) * fraction_to_use
            qddot += amin

        if N == 1:
            return (q.squeeze(), qdot.squeeze(), qddot.squeeze())

        return (q, qdot, qddot)
    
    def solve_and_get_batch_of_traj(self, x0:tuple[np.ndarray, np.ndarray, np.ndarray], xd:tuple[np.ndarray, np.ndarray, np.ndarray], ruckig_as_warm_start:bool=True, ruckig:bool=False, sqp_max_iter:int=SQP_MAX_ITER, line_search_max_iter:int=LINE_SEARCH_MAX_ITER) -> tuple[np.ndarray, list[dict]]:
        """
        With default arguments, solves the polympc problem using ruckig as a warm start for a batch of initial conditions.
        If x0 and xd contain N boundary conditions, then the output will be a batch of N trajectories.
        _______________________________________________________________________________________________________________
        Input :
            x0 (3)                      :   tuple of np.ndarray that contains (initial positions, initial velocities, initial accelerations). 
            xd (3)                      :   tuple of np.ndarray that contains (final positions, final velocities, final accelerations). 
            *ruckig_as_warm_start (1)   :   bool to choose wheter or not ruckig is used as a warm start for polympc
            *ruckig (1)                 :   bool to choose between the polympc trajectory (ruckig=False) or the ruckig one (ruckig=True)
            *sqp_max_iter (1)           :   int to choose the max number of SQP iterations to use in polympc
            *line_search_max_iter (1)   :   int to choose the max number of line search iterations to use in polympc
        Return :
            info    :   dictionnary with fields "status" (1 if polympc converged, 0 otherwise), "iter" (number of sqp
                        iterations to converge) and "time_to_solve".
        _______________________________________________________________________________________________________________
        """

        # Assert input is a tuple of length 3 (q, qdot, qddot)
        assert isinstance(x0, tuple) and len(x0) == 3, "######## MotionPlanner Error : x0 must be a 3D-tuple ########"
        assert isinstance(xd, tuple) and len(xd) == 3, "######## MotionPlanner Error : xd must be a 3D-tuple ########"

        if len(x0[0].shape) == 1: # Means that x0 contains a single initial state, not a list of initial states -> q in R^NDOF
            assert len(xd[0].shape) == 1, "######## MotionPlanner Error : q0 and qd must have the same number of dimensions ########"
            x0 = tuple([np.array([x0[i]]) for i in range(3)])
            xd = tuple([np.array([xd[i]]) for i in range(3)])
        elif len(x0[0].shape) == 2: # Means that x0 contains a list of initial states -> q in R^(Ntraj x NDOF)
            assert len(xd[0].shape) == 2, "######## MotionPlanner Error : q0 and qd must have the same number of dimensions ########"
            assert xd[0].shape[0] == x0[0].shape[0], "######## MotionPlanner Error : q0 and qd must have the same number of trajectories ########"
        else:
            raise ValueError("######## MotionPlanner Error : x0 and xd must be 1D or 2D np.ndarray ########")

        # Assert that each state has the same number of DOF as the robot
        assert xd[0].shape[-1] == self._robot_utils.NDOF, "######## MotionPlanner Error : qd must have the same number of DOF as the robot ########"
        assert x0[0].shape[-1] == self._robot_utils.NDOF, "######## MotionPlanner Error : q0 must have the same number of DOF as the robot ########"

        batch_of_traj = Trajectory(self._robot_utils) # Create a Trajectory object to store multiple trajectories

        # Get the initial and target states
        q0, qdot0, qddot0 = x0
        qd, qdotd, qddotd = xd

        info = []
        # For each initial and target state, solve the problem, store the trajectory and the info
        for q0_i, qdot0_i, qddot0_i, qd_i, qdotd_i, qddotd_i in zip(q0, qdot0, qddot0, qd, qdotd, qddotd):
            # Set the initial and target state
            self.set_current_state((q0_i, qdot0_i, qddot0_i))
            self.set_target_state((qd_i, qdotd_i, qddotd_i))

            # Solve the problem
            info_i = self.solve(ruckig_as_warm_start=ruckig_as_warm_start, ruckig=ruckig, sqp_max_iter=sqp_max_iter, line_search_max_iter=line_search_max_iter)
            info.append(info_i)

            # Get the trajectory and add it to the batch
            traj_i = self.get_trajectory(ruckig=ruckig)
            batch_of_traj += traj_i

        return batch_of_traj, info

    @property
    def info(self):
        if self._info is not None:
            return self._info
        else:
            raise ValueError("######## MotionPlanner Error : Info is None, you should first call the <solve> method ########")

    @property
    def x0(self):
        return self._x0
    
    @property
    def xd(self):
        return self._xd

    @property
    def robot_utils(self):
        return self._robot_utils
    
    @property
    def robot_model(self):
        return self._robotModel

    @property
    def cons_margins(self):
        return self._cons_margins
#----------------------------------------------------------------------------------------------#

def reshape_traj_from_tupple_to_numpy(traj_tupple):
    t, q, qdot, qddot, tau = traj_tupple

    # Adapt dimensions : Traj are stored as [Ntraj, Npts, NDOF]
    t = np.array([t])

    if q.shape[0] < q.shape[1]:
        q = np.array([np.swapaxes(q, 0, 1)])
        qdot = np.array([np.swapaxes(qdot, 0, 1)])
        qddot = np.array([np.swapaxes(qddot, 0, 1)])
        tau = np.array([np.swapaxes(tau, 0, 1)])
    else:
        q = np.array([q])
        qdot = np.array([qdot])
        qddot = np.array([qddot])
        tau = np.array([tau])

    return t, q, qdot, qddot, tau