import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate as integrate

#--------------------------------------------------------------------------------#

def save_trajectory(trajectory, filename="default.npy"):
    """_________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * get_time_to_solve   :   if true, return the solving time of each trajectory
        * cons_margins        :   list of constraints margins between 0 and 1 (1 = no margin).
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""
    
    with open(filename, 'wb') as f:
        np.savez(f, *[[traj_i] for traj_i in trajectory])

#--------------------------------------------------------------------------------#
    
def load_trajectory(filename="default.npy"):
    """_________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * get_time_to_solve   :   if true, return the solving time of each trajectory
        * cons_margins        :   list of constraints margins between 0 and 1 (1 = no margin).
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""
    
    traj_dict = np.load(filename)

    t = traj_dict['arr_0'].squeeze()
    q = traj_dict['arr_1'].squeeze()
    q_dot = traj_dict['arr_2'].squeeze()
    q_ddot = traj_dict['arr_3'].squeeze()
    tau = traj_dict['arr_4'].squeeze()

    return (t, q, q_dot, q_ddot, tau)

#--------------------------------------------------------------------------------#
    
def plot_trajectory(filename="default.npy", trajectory=None, savefig=False):
    """_________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * get_time_to_solve   :   if true, return the solving time of each trajectory
        * cons_margins        :   list of constraints margins between 0 and 1 (1 = no margin).
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""
    
    if trajectory is None:
        t, q, q_dot, q_ddot, tau = load_trajectory(filename=filename)
    else:
        t, q, q_dot, q_ddot, tau = trajectory

    t = t.T
    q = q.T
    q_dot = q_dot.T
    q_ddot = q_ddot.T
    tau = tau.T

    state_to_plot = {"Pos": [q, "rad"], "Vel": [q_dot, "rad/s"], "Acc": [q_ddot, "rad/s²"], "Torque": [tau, "Nm"], "Power": [np.sum(np.abs(q_dot*tau), axis=1), "W"]}
    
    fig, ax = plt.subplots(len(state_to_plot), 1)
    for i, key in enumerate(state_to_plot.keys()):
        ax[i].plot(t, state_to_plot[key][0])
        ax[i].set_ylabel(key + " [" + state_to_plot[key][1] + "]")
        ax[i].set_xlabel("time [s]")
        ax[i].grid()
    
    plt.tight_layout()

    if savefig:
        plt.savefig(filename[0:-4] + '.png')

    plt.show()

#--------------------------------------------------------------------------------#

def discrete_trapezoidal_integration(t, x_dot, x_0=None):
    """_________________________________________________________________________________________________________________

    INPUT PARAMETERS ---------------------------------------------------------------------------------------------------
        | planner             :   motion planner object
        | x                   :   list of initial states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        | xd                  :   list of target states of the form : (np.array(q), np.array(qdot), np.array(qddot))
        * ruckig_as_ws        :   boolean to initialize MPC solver with ruckig
        * get_time_to_solve   :   if true, return the solving time of each trajectory
        * cons_margins        :   list of constraints margins between 0 and 1 (1 = no margin).
        
    OUTPUT PARAMETERS --------------------------------------------------------------------------------------------------
        | traj                :   list of trajectories, where a trajectory is a tupple (t, q, qdot, qddot, tau)
        * time_to_solve       :   list of solving time, returned if get_time_to_solve=True
    ____________________________________________________________________________________________________________________"""
    
    # Initial value of x: x0 if provided, 0 else
    if x_0 is None:
        x_0 = np.zeros_like(x_dot[:, 0])
    
    # Initialize x with x(0) = x0
    x = np.zeros_like(x_dot)
    x[:, 0] = x_0

    print(x.shape, x_dot.shape, t.shape)
    
    N = len(t)
    # Integration
    for i in range(N-1):
        x[:, i+1] = x[:, i] + (x_dot[:, i]+x_dot[:, i+1]) * (t[i+1] - t[i]) / 2

    return x

#--------------------------------------------------------------------------------#    

if __name__ == "__main__":
    # Load default trajectory
    t, q, q_dot, q_ddot, tau = load_trajectory()
    # Plot original trajectory
    plot_trajectory(trajectory=(t, q, q_dot, q_ddot, tau))
    
    # Integrate acceleration to get velocity
    q_dot_int = discrete_trapezoidal_integration(t, q_ddot, q_dot[:, 0])
    # Integrate velocity to get position
    q_int = discrete_trapezoidal_integration(t, q_dot_int, q[:, 0])
    # Plot integrated trajectory
    plot_trajectory(trajectory=(t, q_int, q_dot_int, q_ddot, tau), savefig=True)

