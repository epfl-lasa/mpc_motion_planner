import numpy as np
import enum

# Joint state limits 
q_max = (2.9671, 2.0943, 2.9671, 2.0943, 2.9671, 2.0943, 3.0543)
q_min = (-2.9671, -2.0943, -2.9671, -2.0943, -2.9671, -2.0943, -3.0543)
v_min = (-1.7104, -1.7104, -1.7453, -2.2689, -2.4434, -3.1416, -3.1416)
v_max = (1.7104, 1.7104, 1.7453, 2.2689, 2.4434, 3.1416, 3.1416)
a_min = (-15, -7.5, -10, -12.5, -15, -20, -20)
a_max = (15, 7.5, 10, 12.5, 15, 20, 20)

X_limits = np.array(
    [(q_min[0], q_max[0]), (q_min[1], q_max[1]), (q_min[2], q_max[2]), (q_min[3], q_max[3]),
        (q_min[4], q_max[4]), (q_min[5], q_max[5]), (q_min[6], q_max[6])])  # dimensions of Position
V_limits = np.array(
    [(v_min[0], v_max[0]), (v_min[1], v_max[1]), (v_min[2], v_max[2]), (v_min[3], v_max[3]),
        (v_min[4], v_max[4]), (v_min[5], v_max[5]), (v_min[6], v_max[6])])  # dimensions of Velocity
A_limits = np.array(
    [(a_min[0], a_max[0]), (a_min[1], a_max[1]), (a_min[2], a_max[2]), (a_min[3], a_max[3]),
        (a_min[4], a_max[4]), (a_min[5], a_max[5]), (a_min[6], a_max[6])])


# Define problem parameters
ul = np.array([2.9671, 2.0943, 2.9671, 2.0943, 2.9671, 2.0943, 3.0543]) - 0.25         # Upper limit
ll = np.array([-2.9671, -2.0943, -2.9671, -2.0943, -2.9671, -2.0943, -3.0543]) + 0.25   # Lower limit

CONTROLLED_JOINTS = [1, 2, 3, 4, 5, 6, 7]
NDOF = len(CONTROLLED_JOINTS)

ROBOT_URDF_PATH = "descriptions/robot_descriptions/Kuka_iiwa7_and_14_models/iiwas/model_iiwa7.urdf" #/panda.urdf"
MPC_ROBOT_URDF_PATH = "descriptions/robot_descriptions/Kuka_iiwa7_and_14_models/iiwas/model_iiwa7.urdf" # THIS ONE IS USED ON THE REAL ROBOT
    