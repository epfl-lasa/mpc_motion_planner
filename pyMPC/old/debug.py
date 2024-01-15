import numpy as np
from pyMPC.motion_planner import *
import matplotlib.pyplot as plt
from descriptions.robot_descriptions.franka_panda_bullet.franka_panda import *
import time

q0 = np.array([ 0.67373054,  1.22464762, -0.41799474, -0.9453,     -0.69665789,  1.42159091, -0.990211  ])
dq0 = np.array([-0.62467405, -0.36846654, -0.64362514,  0.84445767, -0.59795651,  1.08140881,  0.        ])

qd = 1.2 * q0
dqd = 0.8 * dq0

planner = MotionPlanner(RobotModel.Panda)
planner.set_current_state((q0, dq0, np.zeros_like(q0)))
planner.set_target_state((qd, dqd, np.zeros_like(qd)))
info = planner.solve()
traj = planner.get_trajectory()
print(traj.q_cons_satisfied)
traj += traj
print(traj.q[0][1] - q0, traj.q[0][-1] - qd)
print(traj.q_cons_satisfied)
print("tau : ", traj.tau_cons_satisfied)

print("slice : ", traj.shape)

# ee_pos, ee_vel = planner.__forward_kinematics(traj.q[1], traj.qdot[1])
# print(ee_pos.shape, ee_vel.shape)

# axis = (1, 2)
# plt.figure()
# plt.plot(ee_pos[-1, axis[0]], ee_pos[-1, axis[1]], 'rx')
# plt.plot(ee_pos[0, axis[0]], ee_pos[0, axis[1]], 'bx')
# plt.plot(ee_pos[:, axis[0]], ee_pos[:, axis[1]])
# plt.quiver(ee_pos[:, axis[0]], ee_pos[:, axis[1]], ee_vel[:, axis[0]], ee_vel[:, axis[1]])
# plt.show()


N = 100_000
traj_list = []
which_cons = []
cond = []

qdot0, qddot = np.zeros((NDOF,)), np.zeros((NDOF,))
dt = 0
count = 0
excluded = 0
qdotd = dqd
safety_range_pos = (1-CONS_MARGINS[0])*(X_limits[:, 1]-X_limits[:, 0])/2
for i in range(N):
    q0 = (X_limits[:, 1]-safety_range_pos - X_limits[:, 0] - safety_range_pos) * np.random.sample((NDOF,)) + X_limits[:, 0] + safety_range_pos
    planner.set_current_state((q0, qdot0, qddot))
    
    while(True):
        print("sample new state")
        old_qdotd = qdotd
        qd = (X_limits[:, 1]-safety_range_pos - X_limits[:, 0] - safety_range_pos) * np.random.sample((NDOF,)) + X_limits[:, 0] + safety_range_pos
        
        qdotd = (V_limits[:, 1] - V_limits[:, 0]) * CONS_MARGINS[1] * (np.random.sample((NDOF,))-0.5) + np.mean(V_limits, axis=1)
        print("qdot", qdotd)
        t0 = time.time()
        # THIS METRIC DOES NOT NEED THE TRAJECTORY TO KNOW IF IT WILL VIOLATE THE CONTRAINTS
        # 7us to compute for 1 traj -> 7ms to compute for 1000 traj
        #Delta = A_limits[:, 1] ** 2 + 2 * J_limits * qdotd
        #left = (np.ones((7,))/(3*J_limits**2) * (A_limits[:, 1] + np.sqrt(Delta))**2 * (3*np.sqrt(Delta) - A_limits[:, 1]))
        
        
        
        # tf = (-A_limits[:, 1] + np.sqrt(Delta)) / J_limits
        # left = 0.5 * A_limits[:, 1] * tf**2 + (1/6) * J_limits * tf**3
        # right = np.min([qd - X_limits[:, 0], X_limits[:, 1] - qd], axis=0)
        # print(tf)
        # dt += time.time() - t0
        # count+=1
        # if np.sum(left < right) == 7: # Le 0.3 est juste un hyper parameter -> 1% des contraints violées avec ca
        #     break
        # excluded += 1

        
        # Stupide metric, on regarde juste si la distance entre le joint et la limite est plus grande qu'un facteur * qdotd
        #if np.sum(np.max([np.abs(X_limits[:, 0] - qd), np.abs(X_limits[:, 1] - qd)], axis=0) > 2 * qdotd) == 7:
        #    break
        
        t0 = time.time()
        t1 = np.abs(qdotd) / (CONS_MARGINS[2] * A_limits[:, 1]) - 1 / 2 / (CONS_MARGINS[4] * J_limits)
        tf = t1 + (CONS_MARGINS[2]) * A_limits[:, 1] / (CONS_MARGINS[4] * J_limits)
        D = 0.5 * (CONS_MARGINS[2]) * A_limits[:, 1] * tf**2 - (CONS_MARGINS[2]) * A_limits[:, 1]**3 / 6 / ((CONS_MARGINS[4]) * J_limits)**2
        #print("D : {} | t1 : {} | tf : {}".format(D, t1, tf))
        dt += (time.time() - t0)
        count += 1
        if np.sum(D < np.min([np.abs(X_limits[:, 1] - qd), np.abs(X_limits[:, 0] - qd)], axis=0)) == 7:
            break
        excluded+=1
        
        

        
        
        

    planner.set_target_state((qd, qdotd, qddot))
    #time.sleep(0.001)
    info = planner.solve(ruckig=True)
    traj = planner.get_trajectory(ruckig=True)
    
    t0 = time.time()
    q_cons, which = traj.q_cons_satisfied
    dt1 = time.time() - t0
    if not(q_cons):
        if len(traj_list) != 0:
            if np.sum(np.abs(traj_list[-1].q[0, -1] - traj.q[0, -1])) > 0:
                traj_list.append(traj)
                which_cons.append(np.where(which)[1])
                print(which_cons[-1])
                t0 = time.time()
                #Delta = A_limits[:, 1] ** 2 + 2 * J_limits * qdotd
                #left = (np.ones((7,))/(3*J_limits**2) * (A_limits[:, 1] + np.sqrt(Delta))**2 * (3*np.sqrt(Delta) - A_limits[:, 1]))
                Delta = 2 * J_limits * qdotd
                left = (np.ones((7,))/(3*J_limits**2) * (np.sqrt(Delta))**2 * (3*np.sqrt(Delta)))
                right = np.abs(qd - X_limits[:, 0])
                dt2 = time.time() - t0
                cond.append([left, right])
                #print(dt1, dt2)
        else:
            traj_list.append(traj)
            which_cons.append(np.where(which)[1])
            print(which_cons[-1])
            t0 = time.time()
            #Delta = A_limits[:, 1] ** 2 + 2 * J_limits * qdotd
            #left = (np.ones((7,))/(3*J_limits**2) * (A_limits[:, 1] + np.sqrt(Delta))**2 * (3*np.sqrt(Delta) - A_limits[:, 1]))
            Delta = 2 * J_limits * qdotd
            left = (np.ones((7,))/(3*J_limits**2) * (np.sqrt(Delta))**2 * (3*np.sqrt(Delta)))
            right = np.abs(qd - X_limits[:, 0])
            dt2 = time.time() - t0
            cond.append([left, right])
        
        #cond_ruckig.append(qdotd - np.sqrt(2 * A_limits[:, 1] * np.max(np.array([np.abs(X_limits[:, 1] - qd), np.abs(X_limits[:, 0] - qd)]), axis=0)))

    print(i+1)






print("dt mean : ", dt/count)
print("excluded : {}".format(excluded))
print("Nb of violated cons : ", len(traj_list))
for k in range(len(traj_list)):
    plt.figure()
    plt.grid()
    p = plt.plot(traj_list[k].t[0], traj_list[k].q[0, :, 0], 
             traj_list[k].t[0], traj_list[k].q[0, :, 1],
             traj_list[k].t[0], traj_list[k].q[0, :, 2],
             traj_list[k].t[0], traj_list[k].q[0, :, 3],
             traj_list[k].t[0], traj_list[k].q[0, :, 4],
             traj_list[k].t[0], traj_list[k].q[0, :, 5],
             traj_list[k].t[0], traj_list[k].q[0, :, 6])
    for idx_cons in which_cons[k]:
        plt.plot(traj_list[k].t[0], np.ones_like(traj_list[k].t[0]) * X_limits[idx_cons, 0], '--', color=p[idx_cons].get_color())
        plt.plot(traj_list[k].t[0], np.ones_like(traj_list[k].t[0]) * X_limits[idx_cons, 1], '--', color=p[idx_cons].get_color())
        plt.plot(traj_list[k].t[0], traj_list[k].qdot[0, :, idx_cons], '.', color=p[idx_cons].get_color())
        plt.plot(traj_list[k].t[0], traj_list[k].qddot[0, :, idx_cons], '-.', color=p[idx_cons].get_color())
    print("qdot : ", traj_list[k].qdot[0, -1])
    print("q : ", traj_list[k].q[0, -1])
    plt.legend(["q{}".format(j+1) for j in range(NDOF)])
    #print("Condition ruckig : ", cond_ruckig[k])
    #print(np.max(np.array([np.abs(X_limits[:, 1] - qd), np.abs(X_limits[:, 0] - qd)]), axis=0))
    #print(cond[k][0] , cond[k][1])
    #cons_viol_pred = np.where(cond[k][0] - cond[k][1] > 0)[0] + 1
    #print("Prediction : ", cons_viol_pred)
    plt.title("cons {} violated".format(which_cons[k]+1))
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")

    
    plt.show()

# q_max = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
# q_min = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)