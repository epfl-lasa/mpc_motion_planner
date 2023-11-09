import motion_planning_lib as mpl
import numpy as np
import enum, time
import sys

""" CONVENTION :
Trajectories dimensions are always in this order : (Ntraj, Npts, NDOF)
Where Ntraj is the number of different trajectories, Npts is the number of time-steps and NDOF the number of degrees of freedom
"""

import descriptions.robot_descriptions.franka_panda_bullet.franka_panda as panda_utils
import descriptions.robot_descriptions.Kuka_iiwa7_and_14_models.kuka_iiwa_7 as kuka7_utils
import descriptions.robot_descriptions.Kuka_iiwa7_and_14_models.kuka_iiwa_14 as kuka14_utils

class RobotModel(enum.Enum):
    Panda = 1
    Kuka7 = 2
    Kuka14 = 3

CONS_MARGINS = [0.9, 0.9, 0.4, 0.7, 0.1]
LINE_SEARCH_MAX_ITER = 10
SQP_MAX_ITER = 3

#----------------------------------------------------------------------------------------------#

class Trajectory():
    def __init__(self, utils, input_traj=None, ruckig=False):
        self._utils = utils
        if input_traj is not None:
            self._t, self._q, self._qdot, self._qddot, self._tau = reshape_traj_from_tupple_to_numpy(input_traj)
        self._ruckig = ruckig

    def _state_cons_satisfied(self, state:np.ndarray, limits:np.ndarray) -> np.ndarray:
        """
        Check constraints satisfaction for a given state and its limits.
        _______________________________________________________________________________________________________________
        Input :
            state   (Ntraj x NPTS x NDOF)   :   np.ndarray containing the actual state values
            limits  (NDOF, 2)               :   np.ndarray of dim  representing the state min and max acceptable values
        Return :
            state_cons_satisfied    (Ntraj) :   np.ndarray (bool) containing a boolean that indicates
                                                wheter or not constraints are satisfied.
        """
        state_cons_satisfied = np.ndarray(shape=(state.shape[0]))
        for i, traj_i_state in enumerate(state):
            traj_i_state_cons_not_satisfied = np.logical_or(traj_i_state > limits[:, 1], traj_i_state < limits[:, 0])
            traj_i_state_cons_not_satisfied = np.sum(traj_i_state_cons_not_satisfied, axis=-1)
            state_cons_satisfied[i] = np.sum(traj_i_state_cons_not_satisfied) == 0
        return state_cons_satisfied

    def _q_cons_satisfied(self) -> np.ndarray:
        """
        Check constraints satisfaction for joint positions.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint position constraints are satisfied.
        """
        return self._state_cons_satisfied(self._q, self._utils.X_limits)

    def _qdot_cons_satisfied(self) -> np.ndarray:
        """
        Check constraints satisfaction for joint velocities.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint velocity constraints are satisfied.
        """
        return self._state_cons_satisfied(self._qdot, self._utils.V_limits)

    def _qddot_cons_satisfied(self) -> np.ndarray:
        """
        Check constraints satisfaction for joint accelerations.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint acceleration constraints are satisfied.
        """
        return self._state_cons_satisfied(self._qddot, self._utils.A_limits)

    def _tau_cons_satisfied(self) -> np.ndarray:
        """
        Check constraints satisfaction for joint torques.
        _______________________________________________________________________________________________________________
        Input :
            None
        Return :
            _state_cons_satisfied    (Ntraj) :  np.ndarray (bool) containing a boolean that indicates
                                                wheter or not joint torque constraints are satisfied.
        """
        T_limits = np.ndarray(shape=(self._tau.shape[-1], 2))
        T_limits[:, 0] = -self._utils.T_limits
        T_limits[:, 1] = self._utils.T_limits
        return self._state_cons_satisfied(self._tau, T_limits)

    def _set_t(self, t:np.ndarray) -> None:
        """
        Set value of the time vector.
        _______________________________________________________________________________________________________________
        Input :
            t   (Ntraj, NPTS)  :    np.ndarray (float) containing the time steps for each trajectory
        Return :
            None
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
        """
        assert isinstance(other, Trajectory)
        assert(self._t.shape[1] == other._t.shape[1])
        t = np.concatenate((self._t, other.t), axis=0)
        q = np.concatenate((self._q, other.q), axis=0)
        qdot = np.concatenate((self._qdot, other.qdot), axis=0)
        qddot = np.concatenate((self._qddot, other.qddot), axis=0)
        tau = np.concatenate((self._tau, other.tau), axis=0)

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
        return self._q_cons_satisfied()

    @property
    def qdot_cons_satisfied(self):
        return self._qdot_cons_satisfied()

    @property
    def qddot_cons_satisfied(self):
        return self._qddot_cons_satisfied()
    
    @property
    def tau_cons_satisfied(self):
        return self._tau_cons_satisfied()
    
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
    def shape(self):
        return self._t.shape

#----------------------------------------------------------------------------------------------#

class MotionPlanner():
    def __init__(self, robotModel):
        if robotModel == RobotModel.Panda:
            self._robot_utils = panda_utils
            self._motion_planner = mpl.PandaMotionPlanner(self._robot_utils.MPC_ROBOT_URDF_PATH)
        elif robotModel == RobotModel.Kuka7:
            self._robot_utils = kuka7_utils
            self._motion_planner = mpl.Kuka7MotionPlanner(self._robot_utils.MPC_ROBOT_URDF_PATH)
        elif robotModel == RobotModel.Kuka14:
            self._robot_utils = kuka14_utils
            #print(self._robot_utils.MPC_ROBOT_URDF_PATH)
            self._motion_planner = mpl.Kuka14MotionPlanner(self._robot_utils.MPC_ROBOT_URDF_PATH)

        self._x0, self._xd = None, None
        self._info, self._cons_margins = None, CONS_MARGINS

    def set_target_state(self, xd:tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Set the target state xd = (q, dq/dt, d²q/dt²) for the motion planner to reach.
        _______________________________________________________________________________________________________________
        Input :
            xd  (3) :   tuple of np.ndarary that contains the desired joint positions, velocities and accelerations.
        Return :
            None
        """
        assert(len(xd) == 3)
        for xd_i in xd:
            assert(len(xd_i.shape) == 1)
            assert(xd_i.shape[0] == self._robot_utils.NDOF)
        
        self._xd = xd
        self._motion_planner.set_target_state(xd[0], xd[1], xd[2])

    def set_current_state(self, x0:tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Set the current state x0 = (q, dq/dt, d²q/dt²) for the motion planner to start from.
        _______________________________________________________________________________________________________________
        Input :
            x0  (3) :   tuple of np.ndarary that contains the current joint positions, velocities and accelerations.
        Return :
            None
        """
        assert(len(x0) == 3)
        for x0_i in x0:
            assert(len(x0_i.shape) == 1)
            assert(x0_i.shape[0] == self._robot_utils.NDOF)
        
        self._x0 = x0
        self._motion_planner.set_current_state(x0[0], x0[1], x0[2])

    def get_trajectory(self, ruckig:bool=False) -> Trajectory:
        """
        Get either the trajectory solved using polympc or the trajectory solved with ruckig (if ruckig=True). A
        trajectory must be available and therefore you should have called the .solve() method at least once before.
        _______________________________________________________________________________________________________________
        Input :
            *ruckig  (1) :  bool to choose between the polympc trajectory (ruckig=False) or the ruckig one (ruckig=True)
        Return :
            traj    : Trajectory object that contains the desired trajectory
        """
        if self._info is None:
            raise RuntimeError("You must have called the .solve method before calling the .get_trajectory one")

        if ruckig:
            traj = Trajectory(self._robot_utils, input_traj=self._motion_planner.get_ruckig_trajectory(), ruckig=True)
        else:        
            traj = Trajectory(self._robot_utils, input_traj=self._motion_planner.get_MPC_trajectory(), ruckig=False)
        
        return traj

    def solve(self, ruckig_as_warm_start:bool=True, ruckig:bool=False, sqp_max_iter:int=SQP_MAX_ITER, line_search_max_iter:int=LINE_SEARCH_MAX_ITER) -> dict:
        """
        With default arguments, solves the polympc problem using ruckig as a warm start. If ruckig=True, the generated
        trajectory does only come from ruckig.
        _______________________________________________________________________________________________________________
        Input :
            *ruckig_as_warm_state (1)   :   bool to choose wheter or not ruckig is used as a warm start for polympc
            *ruckig (1)                 :   bool to choose between the polympc trajectory (ruckig=False) or the ruckig one (ruckig=True)
            *sqp_max_iter (1)           :   int to choose the max number of SQP iterations to use in polympc
            *line_search_max_iter (1)   :   int to choose the max number of line search iterations to use in polympc
        Return :
            info    :   dictionnary with fields "status" (1 if polympc converged, 0 otherwise), "iter" (number of sqp
                        iterations to converge) and "time_to_solve".
        """
        if self._x0 is not None and self._xd is not None:
            #print("before cons_margins")
            self._motion_planner.set_constraint_margins(*self._cons_margins)
            if ruckig:
                start = time.time()
                self._motion_planner.solve_ruckig_trajectory()
                time_to_solve = time.time() - start
            else:
                start = time.time()
                #print("just before in py")
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
        """
        self._cons_margins = cons_margins
        self._motion_planner.set_constraints_margins(*cons_margins)
    
    def __forward_kinematics(self, q:np.ndarray, qdot:np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the end-effector position given the joint positions. If the joint velocities are also given,
        this method will also provide the end-effector velocity.
        _______________________________________________________________________________________________________________
        Input :
            q (Npts, NDOF) or (NDOF)        :   np.ndarray representing the joint positions
            *qdot (Npts, NDOF) or (NDOF)    :   np.ndarray (optional) representing the joint velocities
        Return :
            ee_pos (3)  :   np.ndarray that contains the 3D end effector position
            ee_vel (3)  :   np.ndarray that contains the 3D end effector velocity (if qdot is provided)
        """
        raise NotImplementedError("######## Error in Trajectory.__forward_kinematics : NOT WORKING YET #########")
        assert(len(q.shape) < 3)
        # Compute forward kinematics for a single pair of (q, qdot)
        if len(q.shape) == 1:
            ee_pos = self._motion_planner.forward_kinematics(q)
            ee_vel = None
            if qdot is not None:
                ee_vel = self._motion_planner.forward_velocities(q, qdot)

        # Compute forward kinematics for a full trajectory of (q, qdot)
        elif len(q.shape) == 2:
            ee_pos = np.ndarray(shape=(q.shape[0], 3))
            if qdot is not None:
                ee_vel = np.ndarray(shape=(q.shape[0], 3))
                for i, (q_i, qdot_i) in enumerate(zip(q, qdot)):
                    ee_pos[i] = self._motion_planner.forward_kinematics(q_i)
                    ee_vel_temp = self._motion_planner.forward_velocities(q_i, qdot_i)
                    ee_vel[i] = ee_vel_temp[0:3] # Only keep xyz velocity
            else:
                ee_vel = None
                for i, q_i in enumerate(q):
                    ee_pos[i] = self._motion_planner.forward_kinematics(q_i)

        return ee_pos, ee_vel    

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
    
#----------------------------------------------------------------------------------------------#

def reshape_traj_from_tupple_to_numpy(traj_tupple):
    t, q, qdot, qddot, tau = traj_tupple

    # Adapt dimensions : Traj are stored as [Ntraj, Npts, NDOF]
    t = np.array([t])
    q = np.array([np.swapaxes(q, 0, 1)])
    qdot = np.array([np.swapaxes(qdot, 0, 1)])
    qddot = np.array([np.swapaxes(qddot, 0, 1)])
    tau = np.array([np.swapaxes(tau, 0, 1)])

    return t, q, qdot, qddot, tau