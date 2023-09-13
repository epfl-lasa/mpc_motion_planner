import motion_planning_lib as mpl
import pybullet as p
import numpy as np
from numpy import random
from pathlib import Path
from scipy.interpolate import interp1d
import time, enum, pybullet_data
from plot_utils import save_trajectory, plot_trajectory
from descriptions.robot_descriptions.franka_panda_bullet.franka_panda import *

#--------------------------------------------------------------------------------#

SIM_FREQ = 50
ACTION_REPEAT = 20
ROBOT_URDF_PATH = "descriptions/robot_descriptions/franka_panda_bullet/panda.urdf"
MPC_ROBOT_URDF_PATH = "descriptions/robot_descriptions/franka_panda_bullet/panda_arm.urdf" # THIS ONE IS USED ON THE REAL ROBOT
#ROBOT_URDF_PATH = "descriptions/robot_descriptions/Kuka_iiwa7_and_14_models/iiwas/model_iiwa7.urdf" #/panda.urdf"
#MPC_ROBOT_URDF_PATH = "descriptions/robot_descriptions/Kuka_iiwa7_and_14_models/iiwas/model_iiwa7.urdf" # THIS ONE IS USED ON THE REAL ROBOT

#--------------------------------------------------------------------------------#

class CONTROL_MODE(enum.Enum):
    POSITION=1
    TORQUE=2
    POSITION_VELOCITY=3

DEFAULT_CONTROL_MODE = CONTROL_MODE.POSITION_VELOCITY 

#--------------------------------------------------------------------------------#

def setup_pybullet(robot_urdf_path, gui=True):
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
    
    return p, robotId, planeId

#--------------------------------------------------------------------------------#

def setup_motion_planner(robot_urdf_path, q0, q0_dot=None, q0_ddot=None, cons_margins=[0.85, 0.8, 0.1, 0.9, 0.05], min_height=0.):
    mpc_planner = mpl.MotionPlanner(robot_urdf_path)

    if q0_dot is None:
        q0_dot = np.zeros_like(q0)
    if q0_ddot is None:
        q0_ddot = np.zeros_like(q0)

    mpc_planner.set_current_state(q0.reshape(-1, 1), q0_dot.reshape(-1, 1), q0_ddot.reshape(-1, 1))
    mpc_planner.set_constraint_margins(*cons_margins)
    mpc_planner.set_min_height(min_height)

    return mpc_planner

#--------------------------------------------------------------------------------#

def reset_robot_pos(p, robotId, q):
    p.resetJointStatesMultiDof(robotId, CONTROLLED_JOINTS, [[q_i] for q_i in q])
    p.stepSimulation()

#--------------------------------------------------------------------------------#

def apply_action(p, robotId, action, control_mode=DEFAULT_CONTROL_MODE):
    if control_mode==CONTROL_MODE.POSITION:
        #p.resetJointStatesMultiDof(robotId, CONTROLLED_JOINTS, [[q_i] for q_i in action])
        p.setJointMotorControlArray(robotId, CONTROLLED_JOINTS, p.POSITION_CONTROL, targetPositions=action)
    elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
        p.setJointMotorControlArray(robotId, CONTROLLED_JOINTS, p.POSITION_CONTROL, targetPositions=action[0], targetVelocities=action[1])
    elif control_mode==CONTROL_MODE.TORQUE:
        p.setJointMotorControlArray(robotId, CONTROLLED_JOINTS, p.TORQUE_CONTROL, forces=action)
    else:
        raise ValueError("Control mode not implemented yet")

#--------------------------------------------------------------------------------#

def get_trajectory(planner, x, xd, ruckig_as_ws=True, from_ruckig=False, get_time_to_solve=False):
    planner.set_current_state(*x)
    planner.set_target_state(*xd)
    start = time.time()
    planner.solve_trajectory(ruckig_as_ws)
    time_to_solve = time.time() - start

    if from_ruckig:
        t, q, q_dot, q_ddot, tau = planner.get_ruckig_trajectory()
    else:
        t, q, q_dot, q_ddot, tau = planner.get_MPC_trajectory()
    
    if np.sum(np.abs(tau)) < 1e-9:
        raise ValueError("MPC DIDN'T CONVERGED")
    
    if get_time_to_solve:
        return time_to_solve, (t, q, q_dot, q_ddot, tau)
    else:
        return (t, q, q_dot, q_ddot, tau) # ( t, q(t), q_dot(t), q_ddot(t), tau(t) )

#--------------------------------------------------------------------------------#

def get_action(planner, x, xd, ruckig_as_ws=True, control_mode=DEFAULT_CONTROL_MODE):
    t, q, q_dot, q_ddot, tau = get_trajectory(planner, x, xd, ruckig_as_ws=ruckig_as_ws)
    
    if control_mode==CONTROL_MODE.POSITION:
        f = interp1d(t, q, bounds_error=False, kind='cubic')
        return f(1/SIM_FREQ)
    elif control_mode==CONTROL_MODE.TORQUE:
        f = interp1d(t, tau, bounds_error=False, kind='cubic')
        return f(1/SIM_FREQ)
    elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
        f1 = interp1d(t, q, bounds_error=False, kind='cubic')
        f2 = interp1d(t, q_dot, bounds_error=False, kind='cubic')
        return (f1(1/SIM_FREQ), f2(1/SIM_FREQ)) # return a tupple of (qd, qd_dot)
    else:
        raise ValueError("Control mode not implemented yet")

#--------------------------------------------------------------------------------#

def check_constraints(planner):
    pass

#--------------------------------------------------------------------------------#

def get_interp_action(planner, x, xd, ruckig_as_ws=True, control_mode=DEFAULT_CONTROL_MODE):
    t, q, q_dot, q_ddot, tau = get_trajectory(planner, x, xd, ruckig_as_ws=ruckig_as_ws)
    save_trajectory((t, q, q_dot, q_ddot, tau))
    if control_mode==CONTROL_MODE.POSITION:
        return interp1d(t, q, bounds_error=False, kind='cubic', fill_value=q[:, -1])
    elif control_mode==CONTROL_MODE.TORQUE:
        return interp1d(t, tau, bounds_error=False, kind='cubic', fill_value=tau[:, -1]),
    elif control_mode==CONTROL_MODE.POSITION_VELOCITY:
        return (interp1d(t, q, bounds_error=False, kind='cubic', fill_value=q[:, -1]), 
                interp1d(t, q_dot, bounds_error=False, kind='cubic', fill_value=q_dot[:, -1]))#, fill_value=q_dot[:, -1])) # return a tupple of (qd, qd_dot)
    else:
        raise ValueError("Control mode not implemented yet")

#--------------------------------------------------------------------------------#

def get_observation(p, robotId):
    q = []
    q_dot = []
    tau = []
    for joint in CONTROLLED_JOINTS:
        q_i, q_dot_i, _, tau_i = p.getJointState(robotId, joint)
        q.append(q_i)
        q_dot.append(q_dot_i)
        tau.append(tau_i)

    return np.array(q), np.array(q_dot), np.array(tau)

#--------------------------------------------------------------------------------#

def step(p, robotId, action, control_mode=DEFAULT_CONTROL_MODE):
    apply_action(p, robotId, action, control_mode=control_mode)
    p.stepSimulation()

    obs = get_observation(p, robotId)
    return obs

#------------------------------------- MAIN -------------------------------------#

if __name__ == "__main__":
    # initial joint position
    q0 = np.array(
        [-0.61671491, -1.0231266, -1.58928031, -2.25938556, -1.15041877, 1.92997337, 0.03300055])  # 0.5*(ul+ll)
    q0_dot = np.zeros(NDOF)
    q0_ddot = np.zeros(NDOF)
    g = -9.81

    # Initialize pybullet and robot's position
    p, robotId, _ = setup_pybullet(ROBOT_URDF_PATH, gui=True)
    reset_robot_pos(p, robotId, q0)
    #apply_action(p, robotId, q0, control_mode=CONTROL_MODE.POSITION)
    
    # Initialize mpc planner
    mpc_planner = setup_motion_planner(MPC_ROBOT_URDF_PATH, q0, q0_dot)
    time.sleep(1)

    # sample target point
    qd = (ul - ll) * random.sample(NDOF) + ll
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

    action = get_interp_action(mpc_planner, x_i, xd, control_mode=DEFAULT_CONTROL_MODE, ruckig_as_ws=True)
    status, nb_iter = mpc_planner.get_mpc_info()
    print("status : {} | nb iter : {}".format(status, nb_iter))
    time.sleep(1)

    t = 0
    while(error > tol):
        loop_time = -time.time()
        q_dot_old = q_dot_i # Save old velocity to compute derivative

        t_full_traj.append(t)
        q_full_traj.append(q_i)
        q_dot_full_traj.append(q_dot_i)
        q_ddot_full_traj.append(q_ddot_i)
        tau_full_traj.append(tau_i)

        # one step forward in simulation
        if DEFAULT_CONTROL_MODE==CONTROL_MODE.POSITION_VELOCITY:
            q_i, q_dot_i, tau_i = step(p, robotId, (action[0](t+dt), action[1](t+dt)), control_mode=DEFAULT_CONTROL_MODE)
        else:
            q_i, q_dot_i, tau_i = step(p, robotId, action(t+dt), control_mode=DEFAULT_CONTROL_MODE)
        
        # Get acceleration from PyBullet velocities
        q_ddot_i = (q_dot_i - q_dot_old)/dt
        #tau=10e-3
        #q_ddot_i = (2*tau-dt)/(2*tau+dt) * q_ddot_i + 2 / (2*tau + dt) * (q_dot_i - q_dot_old)
        
        t += dt
        loop_time += time.time()
        time.sleep(max(dt - loop_time, dt/100))

        error = np.linalg.norm(q_i - qd)
        print("Time : {:.2f} [sec] | Error : {:.3f} [rad]".format(t, error))

    
    input("End simulation..")
    p.disconnect()

    traj = get_trajectory(mpc_planner, x_0, xd) # Target trajectory from MPC
    plot_trajectory(trajectory=traj, savefig=True)

    traj_ruckig = get_trajectory(mpc_planner, x_0, xd, from_ruckig=True) # Initial guess from ruckig
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