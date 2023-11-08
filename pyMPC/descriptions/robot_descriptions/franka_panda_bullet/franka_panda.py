import numpy as np
import enum

# Joint state limits 
q_max = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
q_min = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
v_min = (-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100)
v_max = (2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100)
a_min = (-15, -7.5, -10, -12.5, -15, -20, -20)
a_max = (15, 7.5, 10, 12.5, 15, 20, 20)
tau_max = (87, 87, 87, 87, 12, 12, 12)

X_limits = np.array(
    [(q_min[0], q_max[0]), (q_min[1], q_max[1]), (q_min[2], q_max[2]), (q_min[3], q_max[3]),
        (q_min[4], q_max[4]), (q_min[5], q_max[5]), (q_min[6], q_max[6])])  # dimensions of Position
V_limits = np.array(
    [(v_min[0], v_max[0]), (v_min[1], v_max[1]), (v_min[2], v_max[2]), (v_min[3], v_max[3]),
        (v_min[4], v_max[4]), (v_min[5], v_max[5]), (v_min[6], v_max[6])])  # dimensions of Velocity
A_limits = np.array(
    [(a_min[0], a_max[0]), (a_min[1], a_max[1]), (a_min[2], a_max[2]), (a_min[3], a_max[3]),
        (a_min[4], a_max[4]), (a_min[5], a_max[5]), (a_min[6], a_max[6])])
T_limits = np.array(tau_max)


# Define problem parameters
ul = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) - 0.25         # Upper limit
ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.25   # Lower limit

MAX_LINEAR_VELOCITY = 1.7

ROBOT_EF_IDX = 11
CONTROLLED_JOINTS = [0, 1, 2, 3, 4, 5, 6]
NDOF = len(CONTROLLED_JOINTS)
EE_LINK_NAME = "panda_tool"

ROBOT_URDF_PATH = "descriptions/robot_descriptions/franka_panda_bullet/panda.urdf"
MPC_ROBOT_URDF_PATH = "descriptions/robot_descriptions/franka_panda_bullet/panda_arm.urdf" # THIS ONE IS USED ON THE REAL ROBOT