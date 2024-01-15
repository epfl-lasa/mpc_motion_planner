import motion_planning_lib as mpl
from .mpc_trajectory_planning import *
from descriptions.robot_descriptions.franka_panda_bullet.franka_panda import *

ROBOT_URDF_PATH = "descriptions/robot_descriptions/franka_panda_bullet/panda.urdf"
MPC_ROBOT_URDF_PATH = "descriptions/robot_descriptions/franka_panda_bullet/panda_arm.urdf" # THIS ONE IS USED ON THE REAL ROBOT

class Metrics:
    def __init__(self, N):
        self._time_to_converge = np.zeros((N))
        self._nb_iterations = np.zeros((N))
        self._solver_status = np.zeros((N))
        self._x0 = np.zeros((NDOF, N))
        self._xd = np.zeros((NDOF, N))
        self._idx_to_fill = 0 # Iteration variable that keeps track of where to fill in the tab

    def _set_x0(self, x0, i=None):    
        self._x0[:, self._idx_to_fill] = x0[0]

    def _set_xd(self, xd):
        self._xd[:, self._idx_to_fill] = xd[0]

    def _set_stats(self, time_to_converge, solver_status, nb_iterations):
        self._time_to_converge[self._idx_to_fill] = time_to_converge
        self._nb_iterations[self._idx_to_fill] = nb_iterations
        self._solver_status[self._idx_to_fill] = solver_status

    def add_datapoint(self, x0, xd, stats):
        self._set_x0(x0)
        self._set_xd(xd)
        self._set_stats(*stats)
        self._idx_to_fill += 1

    @property
    def time_to_converge(self):
        return self._time_to_converge[0:self._idx_to_fill]
    
    @property
    def nb_iterations(self):
        return self._nb_iterations[0:self._idx_to_fill]
    
    @property
    def x0(self):
        return self._x0[0:self._idx_to_fill]
    
    @property
    def xd(self):
        return self._xd[0:self._idx_to_fill]


if __name__=="__main__":
    # initial joint position
    q0 = np.array(
        [-0.61671491, -1.0231266, -1.58928031, -2.25938556, -1.15041877, 1.92997337, 0.03300055])  # 0.5*(ul+ll)
    q0_dot = np.zeros(NDOF)
    q0_ddot = np.zeros(NDOF)

    # Initialize pybullet and robot's position
    p, robotId, _ = setup_pybullet(ROBOT_URDF_PATH, gui=False)

    # Initialize mpc planner
    mpc_planner = setup_motion_planner(MPC_ROBOT_URDF_PATH, q0, q0_dot)
    time.sleep(0.5)

    N = 1000 # Number of trajectories to generate
    motion_planner_metrics = Metrics(N)

    for i in range(N):
        # Sample start states
        q0 = (ul - ll) * random.sample(NDOF) + ll
        q0_dot = np.zeros(NDOF)
        q0_ddot = np.zeros(NDOF)
        x0 = (q0, q0_dot, q0_ddot)
        # Set position
        reset_robot_pos(p, robotId, q0)

        # Sample final states
        qd = (ul - ll) * random.sample(NDOF) + ll
        qd_dot = np.zeros(NDOF)
        qd_ddot = np.zeros(NDOF)
        xd = (qd, qd_dot, qd_ddot)

        # Get trajectory
        dT, traj = get_trajectory(mpc_planner, x0, xd, ruckig_as_ws=True, get_time_to_solve=True)

        # Get metrics and save
        mpc_info = mpc_planner.get_mpc_info()
        stats = (dT, mpc_info[0], mpc_info[1])
        motion_planner_metrics.add_datapoint(x0, xd, stats)

        # Show
        print("Iteration {} | {:.3f} [sec] | status : {} | iter : {}".format(i+1, dT, mpc_info[0], mpc_info[1]))

    time_to_converge = motion_planner_metrics.time_to_converge
    meanT, minT, maxT = time_to_converge.mean(), time_to_converge.min(), time_to_converge.max()
    print("Time [sec] after {} iterations || mean : {:.3f} | min : {:.3f} |max : {:.3f}".format(i, meanT, minT, maxT))