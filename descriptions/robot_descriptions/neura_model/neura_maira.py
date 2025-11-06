import numpy as np
import enum

# Joint state limits 
q_max = (3.14159265, 2.0943951 , 3.14159265, 2.61799388, 3.14159265, 2.53072742, 3.14159265)
q_min = (-3.14159265, -2.0943951 , -3.14159265, -2.61799388, -3.14159265, -2.53072742, -3.14159265)
v_min = (-2.0943951, -2.0943951, -2.0943951, -2.0943951, -2.0943951, -2.0943951, -2.0943951)
v_max = (2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.0943951)
a_min = (-15, -7.5, -10, -12.5, -15, -20, -20) # TODO
a_max = (15, 7.5, 10, 12.5, 15, 20, 20) # TODO
tau_max = (87, 87, 87, 87, 12, 12, 12) # TODO
j_max = (7500, 3750, 5000, 6250, 7500, 10000, 10000) # TODO

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
J_limits = np.array(j_max)


# Define problem parameters
ul = np.array([3.14159265, 2.0943951 , 3.14159265, 2.61799388, 3.14159265, 2.53072742, 3.14159265])      # Upper limit
ll = np.array([-3.14159265, -2.0943951 , -3.14159265, -2.61799388, -3.14159265, -2.53072742, -3.14159265])  # Lower limit

ROBOT_EF_IDX = 11
CONTROLLED_JOINTS = [2, 3, 4, 5, 6, 7, 8]
NDOF = len(CONTROLLED_JOINTS)
EE_LINK_NAME = "maira7M_grasptarget"

ROBOT_URDF_PATH = "descriptions/robot_descriptions/neura_model/maira7M_gripper.urdf"
MPC_ROBOT_URDF_PATH = "descriptions/robot_descriptions/neura_model/maira7M_gripper.urdf" # THIS ONE IS USED ON THE REAL ROBOT