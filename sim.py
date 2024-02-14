import numpy as np
import casadi as ca
from matplotlib import pyplot as plt
from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneOptControl import PlaneOptControl
from data_vis.DataVisualizer import DataVisualizer

plt.close('all')

mpc_params = {
    'N': 20,
    'Q': ca.diag([0.2, 0.2, 0.2, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0, 0, 0, 0.0]),
    'dt': 0.1
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(15),
    'u_theta_max': np.deg2rad(15),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   10,
    'v_cmd_max':   20
}

state_constraints = {
    'x_min': -np.inf,
    'x_max': np.inf,
    'y_min': -np.inf,
    'y_max': np.inf,
    'z_min': -10,
    'z_max': 50,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(15),
    'theta_max': np.deg2rad(15),
    # 'psi_min':  -np.deg2rad(180),
    # 'psi_max':   np.deg2rad(180),
    'airspeed_min': 2,
    'airspeed_max': 10
}

init_states = np.array([0, #x 
                        0, #y
                        0, #z
                        0, #phi
                        0, #theta
                        np.deg2rad(0), #psi# 3  \
                        5 #airspeed
                        ]) 

final_states = np.array([100, #x
                         100, #y
                         5, #z
                         0,  #phi
                         0,  #theta
                         0,  #psi
                         5 #airspeed
                         ]) 

init_controls = np.array([0, 
                          0, 
                          0, 
                          control_constraints['v_cmd_max']])

goal_params = {
    'x': [final_states[0]],
    'y': [final_states[1]],
    'z': [final_states[2]],
    'radii': [0.5]
}   

data_vis = DataVisualizer()
plane = Plane()
plane.set_state_space()


#### SET YOUR CONFIGURATIONS HERE #######
USE_BASIC = False
USE_OBSTACLE = True


############   MPC   #####################
if USE_BASIC:
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane
    )
    
elif USE_OBSTACLE:
    N_obstacles = 5
    obs_x = np.random.uniform(30, 60, N_obstacles)
    obs_y = np.random.uniform(30, 60, N_obstacles)
    random_radii = np.random.uniform(2.0, 4.0, N_obstacles)

    obs_avoid_params = {
        'weight': 1,
        'safe_distance': 1E-1,
        'x': obs_x,
        'y': obs_y,
        'radii': random_radii
    }
    
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane,
        use_obstacle_avoidance=True,
        obs_params=obs_avoid_params
    )
    
plane_mpc.init_optimization_problem()
#solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)

sim_iteration = 5

solution_history = []
idx_next_step = -1
for i in range(sim_iteration):
    solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)
    
    #next states
    next_x = solution_results['x'][idx_next_step]
    next_y = solution_results['y'][idx_next_step]
    next_z = solution_results['z'][idx_next_step]
    next_phi = solution_results['phi'][idx_next_step]
    next_theta = solution_results['theta'][idx_next_step]
    next_psi = solution_results['psi'][idx_next_step]
    next_v = solution_results['v'][idx_next_step]
    init_states = np.array([next_x, next_y, next_z, next_phi, next_theta, next_psi, next_v])
    
    #next controls
    next_u_phi = solution_results['u_phi'][idx_next_step]
    next_u_theta = solution_results['u_theta'][idx_next_step]
    next_u_psi = solution_results['u_psi'][idx_next_step]
    next_v_cmd = solution_results['v_cmd'][idx_next_step]
    init_controls = np.array([next_u_phi, next_u_theta, next_u_psi, next_v_cmd])
    
    print("init_states: ", init_states)
    print("init_controls: ", init_controls)    
    solution_history.append(solution_results)
    
entire_solution = data_vis.unpack_solution_list(solution_history)

### Plot the solution
fig,ax = data_vis.plot_trajectory_3d(entire_solution)
#plot the goal location
ax.scatter(final_states[0], final_states[1], final_states[2], color='red', s=100, 
           label='Goal')  # Point in red
data_vis.plot_obstacles_3D(goal_params, ax, z_low=-5, z_high=5, color_obs='green')
ax.legend()

if USE_OBSTACLE:
    #plot the obstacles
    for i in range(N_obstacles):
        obstacle = {
            'x': [obs_x[i]],
            'y': [obs_y[i]],
            'z': [0],
            'radii': [random_radii[i]]
        }
        data_vis.plot_obstacles_3D(obstacle, ax, z_low=-5, z_high=5, color_obs='red')
        

### animate the solution
fig,ax, animate = data_vis.animate_trajectory_3D(entire_solution)
ax.scatter(final_states[0], final_states[1], final_states[2], color='red', s=100,
              label='Goal')  # Point in red
data_vis.plot_obstacles_3D(goal_params, ax, z_low=-5, z_high=5, color_obs='green')

ax.legend()

plt.show()