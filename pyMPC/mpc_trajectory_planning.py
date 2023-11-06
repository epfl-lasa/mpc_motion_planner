import motion_planning_lib as mpl
import pybullet as p
import numpy as np
from numpy import random
from pathlib import Path
from scipy.interpolate import interp1d
import time, enum, pybullet_data
import sys, os
#import multiprocessing as mp

from multiprocessing import Process
from multiprocessing.managers import BaseManager

#from pathos.multiprocessing import ProcessingPool as Pool
#from pathos.pools import ProcessPool as Pool
#print(os.getcwd())

from .plot_utils import save_trajectory, plot_trajectory


#sys.path.append(os.path.join(os.getcwd(), '../'))
sys.path.append("/home/stephen/Desktop/LASA/epfl-lasa/mpc_motion_planner/")

#-----------------------------------------------------------------------------------------------------------------------#

NB_TRAJ = 5
ROBOT_MODEL = "PANDA"
NPTS = 100 # Number of points returned by the get_MPC_trajectory function
SIM_FREQ = 50
ACTION_REPEAT = 20
#CONS_MARGINS = [0.85, 0.8, 0.1, 0.9, 0.05] #(old)

#CONS_MARGINS = [0.95, 0.95, 0.3, 0.9, 0.05] #(yang)
CONS_MARGINS = [0.9, 0.9, 0.4, 0.7, 0.1]

#CONS_MARGINS = [0.95, 0.95, 0.3, 0.9, 0.1] #(TRUE yang)
#CONS_MARGINS = [0.85, 0.8, 0.1, 0.9, 0.1] #(albéric)

if ROBOT_MODEL=="PANDA":
    from descriptions.robot_descriptions.franka_panda_bullet.franka_panda import *
    #print(ROBOT_URDF_PATH)
    #print(MPC_ROBOT_URDF_PATH)
elif ROBOT_MODEL=="KUKA7":
    from descriptions.robot_descriptions.Kuka_iiwa7_and_14_models.kuka_iiwa_7 import *
elif ROBOT_MODEL=="KUKA14":
    from descriptions.robot_descriptions.Kuka_iiwa7_and_14_models.kuka_iiwa_14 import *

#-----------------------------------------------------------------------------------------------------------------------#

class CONTROL_MODE(enum.Enum):
    POSITION=1
    TORQUE=2
    POSITION_VELOCITY=3

DEFAULT_CONTROL_MODE = CONTROL_MODE.POSITION_VELOCITY

#-----------------------------------------------------------------------------------------------------------------------#

class CustomManager(BaseManager):
    pass

def setup_pybullet(robot_urdf_path=ROBOT_URDF_PATH, gui=True):
    """
    Setup pybullet environment and return the robot identifier to apply actions to
    ____________________________________________________________________________________________________________________
    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        * robot_urdf_path   :   relative path to the urdf file describing the robot
        * gui               :   boolean to dis/enable pybullet GUI
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | robotId           :   robot identifier in pybullet
        | planeId           :   plane identifier in pybullet
    ____________________________________________________________________________________________________________________"""
    
    if gui:
        clid = p.connect(p.GUI) # Connect and set viz parameters
    else:
        clid = p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=-40+200, cameraPitch=-5, cameraTargetPosition=[0.3, 0.3, 0.6])
    g = -9.81
    p.setGravity(0, 0, g)

    # Sim frequency
    delta_t = 1.0 / SIM_FREQ
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)

    # Load plan, robot and change dynamics
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    planeId = p.loadURDF("plane.urdf", [0, 0, -3.0])
    p.changeDynamics(planeId, -1, restitution=0.9)
    robotId = p.loadURDF(robot_urdf_path, [0, 0, 0 + 0.6], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    p.changeDynamics(robotId, 9, jointUpperLimit=100)
    p.changeDynamics(robotId, 10, jointUpperLimit=100)
    
    return robotId, planeId

#-----------------------------------------------------------------------------------------------------------------------#

def setup_motion_planner(robotModel, q0=None, q0_dot=None, q0_ddot=None, robot_urdf_path=None, cons_margins=CONS_MARGINS, min_height=0., opt=False):
    """
    Setup the motion planner, initial conditions, contraints margins and minimum height. Return the corresponding 
    motion planner object.
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | robotModel          :   string describing the robot model : "PANDA" , "KUKA7", "KUKA14"
        | q0                  :   initial joint configuration as numpy array
        * q0_dot              :   initial joint velocity as numpy array
        * q0_ddot             :   initial joint acceleration as numpy array
        * robot_urdf_path     :   relative path to the urdf file describing the robot
        * cons_margins        :   list of constraints margins between 0 and 1 (1 = no margin).
        * min_height          :   height limit for robot's end effector (?)
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | mpc_planner         :   Motion Planner object
    ____________________________________________________________________________________________________________________"""
    
    print(os.getcwd())

    if robotModel=="PANDA":
        if robot_urdf_path is None:
            #print(MPC_ROBOT_URDF_PATH)
            if opt:
                CustomManager.register('PandaMotionPlanner', mpl.PandaMotionPlanner)
                with CustomManager() as manager:
                    mpc_planner = manager.PandaMotionPlanner(MPC_ROBOT_URDF_PATH)
            else:
                mpc_planner = mpl.PandaMotionPlanner(MPC_ROBOT_URDF_PATH)
        else:
            #print(robot_urdf_path)
            mpc_planner = mpl.PandaMotionPlanner(robot_urdf_path)

    elif robotModel=="KUKA7":
        if robot_urdf_path is None:
            mpc_planner = mpl.Kuka7MotionPlanner(ROBOT_URDF_PATH)
        else:
            mpc_planner = mpl.Kuka7MotionPlanner(robot_urdf_path)
    elif robotModel=="KUKA14":
        if robot_urdf_path is None:
            mpc_planner = mpl.Kuka14MotionPlanner(ROBOT_URDF_PATH)
        else:
            mpc_planner = mpl.Kuka14MotionPlanner(robot_urdf_path)

    if q0 is not None:
        if q0_dot is None:
            q0_dot = np.zeros_like(q0)
        if q0_ddot is None:
            q0_ddot = np.zeros_like(q0)

        mpc_planner.set_current_state(q0.reshape(-1, 1), q0_dot.reshape(-1, 1), q0_ddot.reshape(-1, 1))

    mpc_planner.set_constraint_margins(*cons_margins)
    mpc_planner.set_min_height(min_height)

    return mpc_planner

#-----------------------------------------------------------------------------------------------------------------------#

def reset_robot_pos(robotId, q):
    """
    Reset the robot position to q
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | robotId   :   robot identifier in pybullet
        | q         :   desired reset position
    ____________________________________________________________________________________________________________________"""
    
    p.resetJointStatesMultiDof(robotId, CONTROLLED_JOINTS, [[q_i] for q_i in q])
    p.stepSimulation()

#-----------------------------------------------------------------------------------------------------------------------#

def apply_action(robotId, action, control_mode=DEFAULT_CONTROL_MODE):
    """
    Apply action to the robot, depending on the control_mode.
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | robotId       :   robot identifier in pybullet
        | action        :   for position control, np.array containing target joint positions.
                            for PD control, tupple of np.array containing target joint positions and velocities.
        * control_mode  :   choose between position, PD or torque (torque not working yet)
    ____________________________________________________________________________________________________________________"""

    if control_mode==CONTROL_MODE.POSITION:
        p.setJointMotorControlArray(robotId, CONTROLLED_JOINTS, p.POSITION_CONTROL, targetPositions=action)
    elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
        p.setJointMotorControlArray(robotId, CONTROLLED_JOINTS, p.POSITION_CONTROL, targetPositions=action[0], targetVelocities=action[1])
    elif control_mode==CONTROL_MODE.TORQUE:
        p.setJointMotorControlArray(robotId, CONTROLLED_JOINTS, p.TORQUE_CONTROL, forces=action)
    else:
        raise ValueError("Control mode not implemented yet")

#-----------------------------------------------------------------------------------------------------------------------#

def get_trajectory_list(planner, x, xd, ruckig_as_ws=True, from_ruckig=False, get_time_to_solve=False):
    """
    Return N trajectories from the planner, corresponding to N boundary conditions (x, xd).
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * from_ruckig         :   boolean to get only the ruckig trajectory
        * get_time_to_solve   :   if true, return the solving time of each trajectory
        * cons_margins        :   list of constraints margins between 0 and 1 (1 = no margin).
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""

    if not(isinstance(x, list)): # transform x, xd into list if not already
        x = [x]
        xd = [xd]

    assert len(x[0]) == len(xd[0])

    traj = []
    time_to_solve = []
    for i in range(len(x)):
        
        planner.set_current_state(*x[i])
        planner.set_target_state(*xd[i])
        start = time.time()
        planner.solve_trajectory(ruckig_as_ws)
        time_to_solve_i = time.time() - start

        if from_ruckig:
            traj_i = planner.get_ruckig_trajectory()
        else:
            traj_i = planner.get_MPC_trajectory()

        traj.append(traj_i)   
        time_to_solve.append(time_to_solve_i) 
    
    if len(traj) == 1: # avoid returning a list when only 1 trajectory is returned
        traj = traj[0]
        time_to_solve = time_to_solve[0]

    if get_time_to_solve:
        return time_to_solve, traj 
    else:
        return traj 

#-----------------------------------------------------------------------------------------------------------------------#

def get_trajectory_numpy(planner, x, xd, ruckig_as_ws=True, from_ruckig=False, ruckig_only=False, sqp_max_iter=3, line_search_max_iter=10):
    """
    Return N trajectories from the planner, corresponding to N boundary conditions (x, xd).
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   tupple of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   tupple of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * from_ruckig         :   boolean to get only the ruckig trajectory
        * get_time_to_solve   :   if true, return the solving time of each trajectory
        * cons_margins        :   list of constraints margins between 0 and 1 (1 = no margin).
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""

    traj, time_to_solve, status, iter = [], [], [], []
    t, q, qdot, qddot, tau = np.ndarray(shape=(0, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1))
    t_ruckig, q_ruckig, qdot_ruckig, qddot_ruckig, tau_ruckig = np.ndarray(shape=(0, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1))
    for i in range(x[0].shape[0]):
        planner.set_current_state(x[0][i], x[1][i], x[2][i])
        planner.set_target_state(xd[0][i], xd[1][i], xd[2][i])
        start = time.time()

        if ruckig_only:
            planner.solve_ruckig_trajectory()
            time_to_solve_i = time.time() - start
            status_i, iter_i = planner.get_mpc_info()
            time_to_solve.append(time_to_solve_i) 
            status.append(status_i)
            iter.append(iter_i)
        else:
            planner.solve_trajectory(ruckig_as_ws, sqp_max_iter, line_search_max_iter) # C'EST ICI QUE CA FOIRE
            time_to_solve_i = time.time() - start
            status_i, iter_i = planner.get_mpc_info()

            traj_i = planner.get_MPC_trajectory()           
            t_i, q_i, qdot_i, qddot_i, tau_i = traj_i
            print(np.swapaxes(t_i.shape, q_i.shape))

            # Time
            t_i = np.reshape(t_i, (1, NPTS+1))
            t = np.append(t, t_i, axis=0)

            # Joint position
            q_i = np.reshape(q_i, (1, NDOF, NPTS+1))
            q = np.append(q, q_i, axis=0)
        
            # Joint velocity
            qdot_i = np.reshape(qdot_i, (1, NDOF, NPTS+1))
            qdot = np.append(qdot, qdot_i, axis=0)

            # Joint acceleration
            qddot_i = np.reshape(qddot_i, (1, NDOF, NPTS+1))
            qddot = np.append(qddot, qddot_i, axis=0)

            # Torque
            tau_i = np.reshape(tau_i, (1, NDOF, NPTS+1))
            tau = np.append(tau, tau_i, axis=0)

            traj.append(traj_i)
            time_to_solve.append(time_to_solve_i) 
            status.append(status_i)
            iter.append(iter_i)

        if from_ruckig or ruckig_only:
            traj_i_ruckig = planner.get_ruckig_trajectory()
            t_i_ruckig, q_i_ruckig, qdot_i_ruckig, qddot_i_ruckig, tau_i_ruckig = traj_i_ruckig

            # Time
            t_i_ruckig = np.reshape(t_i_ruckig, (1, NPTS+1))
            t_ruckig = np.append(t_ruckig, t_i_ruckig, axis=0)

            # Joint position
            q_i_ruckig = np.reshape(q_i_ruckig, (1, NDOF, NPTS+1))
            q_ruckig = np.append(q_ruckig, q_i_ruckig, axis=0)
        
            # Joint velocity
            qdot_i_ruckig = np.reshape(qdot_i_ruckig, (1, NDOF, NPTS+1))
            qdot_ruckig = np.append(qdot_ruckig, qdot_i_ruckig, axis=0)

            # Joint acceleration
            qddot_i_ruckig = np.reshape(qddot_i_ruckig, (1, NDOF, NPTS+1))
            qddot_ruckig = np.append(qddot_ruckig, qddot_i_ruckig, axis=0)

            # Torque
            tau_i_ruckig = np.reshape(tau_i_ruckig, (1, NDOF, NPTS+1))
            tau_ruckig = np.append(tau_ruckig, tau_i_ruckig, axis=0)
    
    if i==0: # avoid returning a list when only 1 trajectory is returned
        time_to_solve = time_to_solve[0]
        status = status[0]
        iter = iter[0]

    

    info = {"status": status, "iter": iter, "time": time_to_solve}
    #print(info)

    if from_ruckig:
        return info, (t, q, qdot, qddot, tau), (t_ruckig, q_ruckig, qdot_ruckig, qddot_ruckig, tau_ruckig)
    elif ruckig_only:
        return info, (t_ruckig, q_ruckig, qdot_ruckig, qddot_ruckig, tau_ruckig)
    else:
        return info, (t, q, qdot, qddot, tau)

#-----------------------------------------------------------------------------------------------------------------------#

def get_trajectory_optimized(planner, x, xd, ruckig_as_ws=True, from_ruckig=False, cons_margins=CONS_MARGINS):
    """
    Get N actions from the planner, corresponding to N bounday conditions (x, xd) using multiprocessing
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * control_mode  :   choose between position, PD or torque (torque not working yet)

        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""
    
    traj, time_to_solve, status, iter = [], [], [], []
    t, q, qdot, qddot, tau = np.ndarray(shape=(0, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1))
    t_ruckig, q_ruckig, qdot_ruckig, qddot_ruckig, tau_ruckig = np.ndarray(shape=(0, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1)), np.ndarray(shape=(0, NDOF, NPTS+1))
    #for i in range(x[0].shape[0]):

    #Ncpu = cpu_count()
    #pool = Process(2)
    #results = []
    #results = pool.map(get_trajectory_numpy, [(planner, xi, xdi) for xi, xdi in zip(x, xd)])#.get()
    #results = [r.get()[0] for r in results_object]
    
    
    

    # processes = [Process(target=get_trajectory_numpy, args=(planner, x, xd))]
    # for process in processes:
    #     process.start()


    # for process in processes:
    #     process.join()

    #print(processes)
    print("DONE")


    if len(traj) == 1: # avoid returning a list when only 1 trajectory is returned
        traj = traj[0]
        time_to_solve = time_to_solve[0]
        status = status[0]
        iter = iter[0]

    info = {"status": status, "iter": iter, "time": time_to_solve}

    if from_ruckig:
        return info, (t, q, qdot, qddot, tau), (t_ruckig, q_ruckig, qdot_ruckig, qddot_ruckig, tau_ruckig)
    else:
        return info, (t, q, qdot, qddot, tau)

#-----------------------------------------------------------------------------------------------------------------------#

def get_action(planner, x, xd, ruckig_as_ws=True, control_mode=DEFAULT_CONTROL_MODE):
    """
    Get N actions from the planner, corresponding to N bounday conditions (x, xd)
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * control_mode  :   choose between position, PD or torque (torque not working yet)

        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""
    
    traj = get_trajectory_list(planner, x, xd, ruckig_as_ws=ruckig_as_ws)

    if not(isinstance(traj, list)): # 
        traj = [traj]

    f, f1, f2 = [], [], [] # Interplation loop
    for traj_i in traj:
        t_i, q_i, q_dot_i, _, tau_i = traj_i
        if control_mode==CONTROL_MODE.POSITION:
            f.append(interp1d(t_i, q_i, bounds_error=False, kind='cubic'))
        elif control_mode==CONTROL_MODE.TORQUE:
            f.append(interp1d(t_i, tau_i, bounds_error=False, kind='cubic'))
        elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
            f1.append(interp1d(t_i, q_i, bounds_error=False, kind='cubic'))
            f2.append(interp1d(t_i, q_dot_i, bounds_error=False, kind='cubic'))
        else:
            raise ValueError("Control mode not implemented yet")
        
    # Return list of interpolations
    if control_mode==CONTROL_MODE.POSITION:
        return f(1/SIM_FREQ)
    elif control_mode==CONTROL_MODE.TORQUE:
        return f(1/SIM_FREQ)
    elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
        return (f1(1/SIM_FREQ), f2(1/SIM_FREQ)) # return a tupple of (qd, qd_dot)

#-----------------------------------------------------------------------------------------------------------------------#

def get_power_cons(trajectory):
    """
    
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------

    ____________________________________________________________________________________________________________________"""
    _, _, qdot, _, tau = trajectory # q, qdot, qddot, tau are [N x NDOF] matrices, where N is the number of trajectories
    
    power_cons = np.ndarray(shape=qdot.shape)
    for i, (qdot_i, tau_i) in enumerate(zip(qdot, tau)):
        power_cons[i] = np.abs(np.multiply(qdot_i, tau_i))

    return power_cons

#-----------------------------------------------------------------------------------------------------------------------#

def kin_dyn_constraints_satisfied(trajectory, ruckig=False):
    """
    
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------

    ____________________________________________________________________________________________________________________"""
    _, q, qdot, qddot, tau = trajectory # q, qdot, qddot, tau are [N x NDOF] matrices, where N is the number of trajectories
    flag = True
    consSatisfied = np.ndarray(shape=(0, 3), dtype=np.bool8)
    idxConsNotSatisfied = np.ndarray(shape=(0, 1), dtype=np.int32)
    for i, (q_i, qdot_i, qddot_i, tau_i) in enumerate(zip(q, qdot, qddot, tau)):
        #print(i, q_i.shape, qdot_i.shape, qddot_i.shape, tau_i.shape)
        kinConsSatisfied_i = kinematic_constraints_satisfied(q_i, qdot_i, qddot_i, ruckig=ruckig)
        dynConsSatisfied_i = dynamic_constraints_satisfied(tau_i)
        all_cons_satisfied = dynConsSatisfied_i[0] and kinConsSatisfied_i[0] # True if all the constraints are satisfied
        consSatisfied = np.append(consSatisfied, np.array([[all_cons_satisfied, kinConsSatisfied_i[0], dynConsSatisfied_i[0]]]), axis=0)
        if not(all_cons_satisfied):
            idxConsNotSatisfied = np.append(idxConsNotSatisfied, np.array([i]))
            #print("kin cons not satisfied : ", kinConsSatisfied_i[1])
            #print("dyn cons not satisfied : ", dynConsSatisfied_i[1])

        #print(idxConsNotSatisfied)
    
    # first value is true if all constraints are always satisfied --> no idx stored
    return idxConsNotSatisfied.shape[0]==0, consSatisfied, idxConsNotSatisfied

#-----------------------------------------------------------------------------------------------------------------------#

def kinematic_constraints_satisfied(q, qdot, qddot, ruckig=False):
    """
    
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------

    ____________________________________________________________________________________________________________________"""
    qViolationIdx, qdotViolationIdx, qddotViolationIdx = [], [], []
    
    for i, (q_i, qdot_i, qddot_i) in enumerate(zip(q.T, qdot.T, qddot.T)): # This loop goes through all the states in ONE trajectory
        if len(q_i.shape) == 1:
            q_i = np.reshape(q_i, (q_i.shape[0], 1))
            qdot_i = np.reshape(qdot_i, (qdot_i.shape[0], 1))
            qddot_i = np.reshape(qddot_i, (qddot_i.shape[0], 1))

        if not(ruckig) and (np.sum(q_i > X_limits[:, 1, None]) or np.sum(q_i < X_limits[:, 0, None])):
            qViolationIdx.append(i)

        if np.sum(qdot_i > V_limits[:, 1, None]) or np.sum(qdot_i < V_limits[:, 0, None]):
            qdotViolationIdx.append(i)

        if np.sum(qddot_i > A_limits[:, 1, None]) or np.sum(qddot_i < A_limits[:, 0, None]):
            qddotViolationIdx.append(i)

    cons_satisfied=True
    if len(qViolationIdx) != 0:
        cons_satisfied=False

    if len(qdotViolationIdx) != 0:
        cons_satisfied=False

    if len(qddotViolationIdx) != 0:
        cons_satisfied=False

    return cons_satisfied, (qViolationIdx, qdotViolationIdx, qddotViolationIdx)

#-----------------------------------------------------------------------------------------------------------------------#

def ee_constraints_satisfied(motion_planner, q, qdot, qddot, ruckig=False):

    eePosViolationIdx, eeVelViolationIdx = [], []
    for i, (q_i, qdot_i, qddot_i) in enumerate(zip(q.T, qdot.T, qddot.T)): # This loop goes through all the states in ONE trajectory
        if len(q_i.shape) == 1:
            q_i = np.reshape(q_i, (q_i.shape[0], 1))
            qdot_i = np.reshape(qdot_i, (qdot_i.shape[0], 1))
            qddot_i = np.reshape(qddot_i, (qddot_i.shape[0], 1))

        ee_pos = motion_planner.forward_kinematics(q_i)
        ee_vel = motion_planner.forward_velocities(q_i, qdot_i, "panda_link7")

        if not(ruckig) and ee_pos[2] < 0.05:
            eePosViolationIdx.append(i)

        if not(ruckig) and np.linalg.norm(ee_vel[0:3]) > 1.7:
            eeVelViolationIdx.append(i)
    
    cons_satisfied = True
    if len(eePosViolationIdx) != 0:
        cons_satisfied=False
    if len(eeVelViolationIdx) != 0:
        cons_satisfied=False

    return cons_satisfied, (eePosViolationIdx, eeVelViolationIdx)

#-----------------------------------------------------------------------------------------------------------------------#

def dynamic_constraints_satisfied(torque):
    """
    
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------

    ____________________________________________________________________________________________________________________"""
    violationIdx = []
    for i, tau_i in enumerate(torque.T):
        if len(tau_i.shape) == 1:
            tau_i = np.reshape(tau_i, (tau_i.shape[0], 1))
        if np.sum(np.abs(tau_i.squeeze()) > T_limits.T):
            violationIdx.append(i)

    if len(violationIdx) == 0:
        return True, violationIdx
    else:
        return False, violationIdx

#-----------------------------------------------------------------------------------------------------------------------#

def compare_trajectories(y1, y2, interp_kind='cubic', discretization_npts=100, criterion="mse"):
    """
    Compare two NxNDOF trajectories. For example x1 and x2 might be two joint position trajectories.
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | y1            :   tupple of (time of dimension [N1,], first trajectory of dimension [N1, NDOF])
        | y2            :   tupple of (time of dimension [N2,], second trajectory of dimension [N2, NDOF])
        * criterion     :   comparison criterion, can be "rmse"
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------

    ____________________________________________________________________________________________________________________"""   
    
    t1, x1 = y1
    t2, x2 = y2

    if len(y1[0].shape) == 1:
        t1 = np.reshape(t1, (1, t1.shape[0]))
        t2 = np.reshape(t2, (1, t2.shape[0]))

    error = np.ndarray(shape=(0))
    for t1_i, x1_i, t2_i, x2_i in zip(t1, x1, t2, x2):
        Tf = np.min([np.max(t1_i), np.max(t2_i)])
        T0 = np.max([np.min(t1_i), np.min(t2_i)])

        # If ValueError: Expect x to not have duplicates --> https://github.com/deepinsight/insightface/issues/2152
        f1 = interp1d(t1_i, x1_i, bounds_error=False, kind=interp_kind)
        f2 = interp1d(t2_i, x2_i, bounds_error=False, kind=interp_kind)

        t = np.linspace(T0, Tf, discretization_npts)

        if criterion=="mse":
            error = np.append(error, np.square(f1(t) - f2(t)).mean())
        else:
            raise ValueError("criterion «" + criterion + "» not implemented yet ! ")

    return error

#-----------------------------------------------------------------------------------------------------------------------#

def compare_discrete_trajectories(x1, x2, criterion="mse"):
    """
    Compare two NxNDOF trajectories. For example x1 and x2 might be two joint position trajectories.
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | x1        :   first trajectory of dimension [N, NDOF]
        | x2        :   second trajectory of dimension [N, NDOF]
        * criterion :   comparison criterion, can be "mse"
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------

    ____________________________________________________________________________________________________________________"""   
    
    if criterion == "mse":
        error = np.square(x1 - x2).mean()
    else:
        raise ValueError("criterion «" + criterion + "» not implemented yet ! ")

    return error
        

#-----------------------------------------------------------------------------------------------------------------------#

def compare_continuous_trajectories(t, x1, x2, criterion="mse"):
    """
    Compare two NxNDOF trajectories. For example x1 and x2 might be two joint position trajectories.
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | x1        :   first trajectory of dimension [N, NDOF]
        | x2        :   second trajectory of dimension [N, NDOF]
        * criterion :   comparison criterion, can be "rmse"
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------

    ____________________________________________________________________________________________________________________"""   
    
    #T = 
    # Discretize

    return error

#-----------------------------------------------------------------------------------------------------------------------#

def get_interp_trajectory(planner, x, xd, ruckig_as_ws=True, from_ruckig=False, control_mode=DEFAULT_CONTROL_MODE):
    """
    Return N trajectories from the planner, corresponding to N boundary conditions (x, xd)
    _________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * from_ruckig         :   boolean to get only the ruckig trajectory
        * control_mode        :   choose between position, PD or torque (torque not working yet)
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        * f                   :   Interpolated function of joint position (position control) or torque (torque control)
        * f1                  :   Interpolated function of joint position (PD control)
        * f2                  :   Interpolated function of joint velocity (PD control)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""
    
    traj = get_trajectory_list(planner, x, xd, from_ruckig=from_ruckig, ruckig_as_ws=ruckig_as_ws)

    if not(isinstance(traj, list)):
        traj = [traj]

    f, f1, f2 = [], [], []
    for traj_i in traj:
        t_i, q_i, q_dot_i, _, tau_i = traj_i
        cons, kinCons, dynCons = kin_dyn_constraints_satisfied(traj_i)
        print("Constraint satisfied ? ", cons)
        if control_mode==CONTROL_MODE.POSITION:
            f.append(interp1d(t_i, q_i, bounds_error=False, kind='cubic', fill_value=q_i[:, -1]))
        elif control_mode==CONTROL_MODE.TORQUE:
            f.append(interp1d(t_i, tau_i, bounds_error=False, kind='cubic', fill_value=tau_i[:, -1]))
        elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
            f1.append(interp1d(t_i, q_i, bounds_error=False, kind='cubic', fill_value=q_i[:, -1]))
            f2.append(interp1d(t_i, q_dot_i, bounds_error=False, kind='cubic', fill_value=q_dot_i[:, -1]))

    # Return interpolation functions
    if control_mode==CONTROL_MODE.POSITION:
        if len(f) == 1:
            f = f[0]
        return f
    elif control_mode==CONTROL_MODE.TORQUE:
        if len(f) == 1:
            f = f[0]
        return f
    elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
        if len(f1) == 1:
            f1 = f1[0]
            f2 = f2[0]
        return (f1, f2)
    else:
        raise ValueError("Control mode not implemented yet")

#-----------------------------------------------------------------------------------------------------------------------#

def get_observation(robotId):
    """
    Return robot's joint states
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | robotId       :   robot identifier in pybullet
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | q             :   observed joint position
        | q_dot         :   observed joint velocity
        | tau           :   oberved torque
    ____________________________________________________________________________________________________________________"""
    
    q = []
    q_dot = []
    tau = []
    for joint in CONTROLLED_JOINTS:
        q_i, q_dot_i, _, tau_i = p.getJointState(robotId, joint)
        q.append(q_i)
        q_dot.append(q_dot_i)
        tau.append(tau_i)

    return np.array(q), np.array(q_dot), np.array(tau)

#-----------------------------------------------------------------------------------------------------------------------#

def step(robotId, action, control_mode=DEFAULT_CONTROL_MODE):
    """
    Apply action to the robot and move one step further in time
    ____________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | robotId             :   robot identifier in pybullet
        | action              :   action to apply to the environment
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * control_mode        :   choose between position, PD or torque (torque not working yet)
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | obs                 :   observed joint position, velocity and torque
    ____________________________________________________________________________________________________________________"""
    
    apply_action(robotId, action, control_mode=control_mode)
    p.stepSimulation()

    obs = get_observation(robotId)
    return obs

#-----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------- MAIN --------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    # initial joint position
    q0 =  0.5*(ul+ll) # np.array([-0.61671491, -1.0231266, -1.58928031, -2.25938556, -1.15041877, 1.92997337, 0.03300055]) 
    q0_dot = np.zeros(NDOF)
    q0_ddot = np.zeros(NDOF)
    g = -9.81

    # Initialize pybullet and robot's position
    robotId, _ = setup_pybullet(gui=True)
    reset_robot_pos(robotId, q0)
    
    # Initialize mpc planner
    mpc_planner = setup_motion_planner(ROBOT_MODEL, q0=q0, q0_dot=q0_dot)
    time.sleep(1)

    # sample target point
    qd = (ul - ll) * random.sample(NDOF) + ll
    #qd = 0.9 * ul + 0.1 * ll
    
    qd_dot = np.zeros(NDOF)
    qd_ddot = np.zeros(NDOF)

    ################# MAIN LOOP ################
    q_i, q_dot_i, q_ddot_i, tau_i = q0, q0_dot, q0_ddot, np.zeros_like(q0)
    x_i = (q_i, q_dot_i, q_ddot_i)
    x_0 = (q0, q0_dot, q0_ddot)
    xd = (qd, qd_dot, qd_ddot)
    tol = 5e-3 # minimal error to end loop
    error = float('inf')
    dt = 1/SIM_FREQ

    t_full_traj = []
    q_full_traj = []
    q_dot_full_traj = []
    q_ddot_full_traj = []
    tau_full_traj = []

    # traj = get_trajectory(mpc_planner, x_0, xd) # Target trajectory from MPC
    # plot_trajectory(trajectory=traj, savefig=True)
    t_traj = 0
    for traj_i in range(NB_TRAJ):
        action = get_interp_trajectory(mpc_planner, x_i, xd, control_mode=DEFAULT_CONTROL_MODE, from_ruckig=False, ruckig_as_ws=True)
        status, nb_iter = mpc_planner.get_mpc_info()
        print("status : {} | nb iter : {}".format(status, nb_iter))
        time.sleep(1)

        t = 0
        while(error > tol):
            loop_time = -time.time()
            q_dot_old = q_dot_i # Save old velocity to compute derivative

            t_full_traj.append(t+t_traj)
            q_full_traj.append(q_i)
            q_dot_full_traj.append(q_dot_i)
            q_ddot_full_traj.append(q_ddot_i)
            tau_full_traj.append(tau_i)

            # one step forward in simulation
            if DEFAULT_CONTROL_MODE==CONTROL_MODE.POSITION_VELOCITY:
                q_i, q_dot_i, tau_i = step(robotId, (action[0](t+dt), action[1](t+dt)), control_mode=DEFAULT_CONTROL_MODE)
            else:
                q_i, q_dot_i, tau_i = step(robotId, action(t+dt), control_mode=DEFAULT_CONTROL_MODE)
            
            # Get acceleration from PyBullet velocities
            q_ddot_i = (q_dot_i - q_dot_old)/dt

            t += dt
            loop_time += time.time()
            time.sleep(max(dt - loop_time, dt/100))

            error = np.linalg.norm(q_i - qd)
            print("Time : {:.2f} [sec] | Error : {:.3f} [rad]".format(t, error))

        t_traj += t
        q_dot_i, q_ddot_i, tau_i = 3*[np.zeros_like(q_i)]
        x_i = (q_i, q_dot_i, q_ddot_i)
        qd = (ul - ll) * random.sample(NDOF) + ll
        #qd = 0.9 * ul + 0.1 * ll
        qd_dot = np.zeros(NDOF)
        qd_ddot = np.zeros(NDOF)
        xd = (qd, qd_dot, qd_ddot)
        error = float('inf')
    
    input("End simulation..")
    p.disconnect()

    traj = get_trajectory_list(mpc_planner, x_0, xd) # Target trajectory from MPC
    plot_trajectory(trajectory=traj, savefig=True)

    traj_ruckig = get_trajectory_list(mpc_planner, x_0, xd, from_ruckig=True) # Initial guess from ruckig
    plot_trajectory(trajectory=traj_ruckig, savefig=True, filename="ruckig.png")

    t_full_traj = np.array(t_full_traj)
    q_full_traj = np.array(q_full_traj).T
    q_dot_full_traj = np.array(q_dot_full_traj).T
    q_ddot_full_traj = np.array(q_ddot_full_traj).T
    tau_full_traj = np.array(tau_full_traj).T
    # Actual trajectory recorded in PyBullet
    plot_trajectory(trajectory=(t_full_traj, q_full_traj, q_dot_full_traj, q_ddot_full_traj, tau_full_traj),
                    savefig=True,
                    filename="effective.png")