import numpy as np
import casadi as ca
import time 
from matplotlib import pyplot as plt
from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneOptControl import PlaneOptControl
from data_vis.DataVisualizer import DataVisualizer

def find_driveby_direction(goal_position:np.ndarray, current_position:np.ndarray, 
                            heading_rad:float, effector_range:float):
    """
    Finds the lateral offset directions of the omnidirectional effector
    """    
    
    ego_unit_vector = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    
    #swap the direction sign to get the normal vector
    drive_by_vector_one = np.array([ego_unit_vector[1], -ego_unit_vector[0]])
    drive_by_vector_two = np.array([-ego_unit_vector[1], ego_unit_vector[0]])
    
    drive_by_vector_one = drive_by_vector_one * effector_range
    drive_by_vector_two = drive_by_vector_two * effector_range
    
    #pick the one closer to the current position
    distance_one = np.linalg.norm(current_position - (goal_position + drive_by_vector_one))
    distance_two = np.linalg.norm(current_position - (goal_position + drive_by_vector_two))
    
    if distance_one < distance_two:
        drive_by_vector = drive_by_vector_one
    else:
        drive_by_vector = drive_by_vector_two
        
    #apply to goal position
    drive_by_position = goal_position + drive_by_vector
    
    return drive_by_position

plt.close('all')

def find_driveby_point(goal_position, current_los):
    pass

###### INITIAL CONFIGURATIONS ########
mpc_params = {
    'N': 30,
    'Q': ca.diag([1E-3, 1E-3, 1E-3, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0.01, 0.01, 0.01, 0.01]),
    'dt': 0.1
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(10),
    'u_theta_max': np.deg2rad(10),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   15,
    'v_cmd_max':   30
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
                        10, #y
                        0, #z
                        0, #phi
                        0, #theta
                        np.deg2rad(70), #psi# 3  \
                        5 #airspeed
                        ]) 

final_states = np.array([250, #x
                         250, #y
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
    'radii': [1.0]
}   

data_vis = DataVisualizer()
plane = Plane()
plane.set_state_space()

#### PLOT CONFIGURATIONS ####
color_obstacle = 'purple'
trajectory_color = 'blue'
goal_color = 'green'

#### SET YOUR CONFIGURATIONS HERE #######
seed = 2
np.random.seed(seed)
sim_iteration = 60
idx_next_step = 5 #index of the next step in the solution
N_obstacles = 9


title_video = 'Omni Directional Effector Obstacle Avoidance'
SAVE_GIF = False

OBX_MIN_RANGE = 75
OBX_MAX_RANGE = 175
OBX_MIN_RADIUS = 5
OBX_MAX_RADIUS = 20

USE_BASIC = False
USE_OBSTACLE = False
USE_TIME_CONSTRAINT = False
USE_DIRECTIONAL_PEW_PEW = False
USE_DIRECTIONAL_PEW_PEW_OBSTACLE = False
USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE = True
############   MPC   #####################
if USE_BASIC:
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params,     
        plane
    )
    
elif USE_OBSTACLE:
    obs_x = np.random.randint(OBX_MIN_RADIUS, OBX_MAX_RANGE, N_obstacles)
    obs_y = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_obstacles)
    random_radii = np.random.randint(OBX_MIN_RADIUS, OBX_MAX_RADIUS, N_obstacles)

    obs_avoid_params = {
        'weight': 1,
        'safe_distance': 3.0,
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
elif USE_TIME_CONSTRAINT:
    mpc_params = {
        'N': 30,
        'Q': ca.diag([1E-2, 1E-2, 1E-2, 0, 0, 0.0, 0.0]),
        'R': ca.diag([0.01, 0.01, 0.01, 0.01]),
        'dt': 0.1
    }
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane,
        use_time_constraints=True,
        time_constraint_val=2.0
    )
elif USE_DIRECTIONAL_PEW_PEW :
    
    mpc_params = {
        'N': 30,
        'Q': ca.diag([1E-3, 1E-3, 1E-3, 0, 0, 0.0, 0.0]),
        'R': ca.diag([0.01, 0.01, 0.01, 0.001]),
        'dt': 0.1
    }
    
    effector_config = {
            'effector_range': 50, 
            'effector_power': 1, 
            'effector_type': 'directional_3d', 
            'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
            'weight': 100, 
            'radius_target': 0.5
            }
    
    obs_avoid_params = {
        'weight': 1E-10,
        'safe_distance': 1.0,
        'x': [],
        'y': [],
        'radii': []
    }
    
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane,
        use_pew_pew=True,
        pew_pew_params=effector_config,
        use_obstacle_avoidance=True,
        obs_params=obs_avoid_params
    )
    
elif USE_DIRECTIONAL_PEW_PEW_OBSTACLE:
    obs_x = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_obstacles)
    obs_y = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_obstacles)
    random_radii = np.random.randint(3, 20, N_obstacles)
    
    
    ## these weights are best for the directional effector
    mpc_params = {
        'N': 30,
        'Q': ca.diag([1E-2, 1E-2, 1E-2, 0, 0, 0.0, 0.0]),
        'R': ca.diag([0.01, 0.01, 0.01, 0.01]),
        'dt': 0.1
    }

    #These weights are not good for the directional effector
    # mpc_params = {
    #     'N': 30,
    #     'Q': ca.diag([1E-1, 1E-1, 1E-1, 0, 0, 0.0, 0.0]),
    #     'R': ca.diag([0.01, 0.01, 0.01, 0.01]),
    #     'dt': 0.1
    # }

    obs_avoid_params = {
        'weight': 0,
        'safe_distance': 3.0,
        'x': obs_x,
        'y': obs_y,
        'radii': random_radii
    }
    
    effector_config = {
            'effector_range': 30, 
            'effector_power': 1, 
            'effector_type': 'directional_3d', 
            'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
            'weight': 1, 
            'radius_target': 2.0
            }
    
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane,
        use_pew_pew=True,
        pew_pew_params=effector_config,
        use_obstacle_avoidance=True,
        obs_params=obs_avoid_params
    )
    
elif USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
    Q_val = 1E-2
    obs_x = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_obstacles)
    obs_y = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_obstacles)
    random_radii = np.random.randint(3, 20, N_obstacles)
    
    obs_z = np.random.randint(0, 1, N_obstacles)
    
    mpc_params = {
        'N': 30,
        'Q': ca.diag([Q_val, Q_val, Q_val, 0, 0, 0.0, 0.0]),
        'R': ca.diag([0.1, 0.1, 0.1, 1.0]),
        'dt': 0.1
    }
    
    effector_config = {
            'effector_range': 10, 
            'effector_power': 1, 
            'effector_type': 'omnidirectional', 
            'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
            'weight': 1, 
            'radius_target': 3.0,
            'minor_radius': 1.0
            }
    
    ## need to set the actual goal position as an obstacle
    
    ## this is very hacky, but we need to set the goal position as an obstacle and 
    ## needs to be the last obstacle
    obs_avoid_params = {
        'weight': Q_val,
        'safe_distance': 1.0,
        'x': obs_x,
        'y': obs_y,
        'z': obs_z,
        'radii': random_radii
    }
    

    obs_avoid_params['x'] = np.append(obs_avoid_params['x'], final_states[0])
    obs_avoid_params['y'] = np.append(obs_avoid_params['y'], final_states[1])
    obs_avoid_params['z'] = np.append(obs_avoid_params['z'], final_states[2])
    obs_avoid_params['radii'] = np.append(obs_avoid_params['radii'], 1.0)
    
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane,
        use_pew_pew=True,
        pew_pew_params=effector_config,
        use_obstacle_avoidance=True,
        obs_params=obs_avoid_params
    )
    
plane_mpc.init_optimization_problem()
#solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)

if USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
    
    driveby_direction = find_driveby_direction(final_states[:2], init_states[:2], 
                                               init_states[5], 
                                               effector_config['effector_range'])
    driveby_states = final_states.copy()
    
    driveby_states[0] = driveby_direction[0]
    driveby_states[1] = driveby_direction[1]
    print("Drive By Direction: ", driveby_direction)

solution_history = []
solution_times = []
for i in range(sim_iteration):
    
    start_time = time.time()
    if USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
        driveby_direction = find_driveby_direction(final_states[:2], init_states[:2], 
                                                init_states[5], 
                                                effector_config['effector_range'])
        driveby_states = final_states.copy()        
        driveby_states[0] = driveby_direction[0]
        driveby_states[1] = driveby_direction[1]
        solution_results = plane_mpc.get_solution(init_states, driveby_states, init_controls)
    else:
        solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)
    final_time = time.time() - start_time
    solution_times.append(final_time)
    
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
    
    #prune out the solution to the next step
    solution_results = {
        'x': solution_results['x'][:idx_next_step],
        'y': solution_results['y'][:idx_next_step],
        'z': solution_results['z'][:idx_next_step],
        'phi': solution_results['phi'][:idx_next_step],
        'theta': solution_results['theta'][:idx_next_step],
        'psi': solution_results['psi'][:idx_next_step],
        'v': solution_results['v'][:idx_next_step],
        'u_phi': solution_results['u_phi'][:idx_next_step],
        'u_theta': solution_results['u_theta'][:idx_next_step],
        'u_psi': solution_results['u_psi'][:idx_next_step],
        'v_cmd': solution_results['v_cmd'][:idx_next_step]
    }
        
    solution_history.append(solution_results)

#%% 
entire_solution = data_vis.unpack_solution_list(solution_history)
#time vector based on length of the solution
time_span = []
for i in range(len(entire_solution['x'])):
    time_span.append(i * mpc_params['dt'])

### Plot the solution
fig,ax = data_vis.plot_trajectory_3d(entire_solution, use_time_color=True, time_list=time_span)
#plot the goal location
ax.scatter(final_states[0], final_states[1], final_states[2], color=goal_color, s=100, 
           label='Goal')  # Point in red
data_vis.plot_obstacles_3D(goal_params, ax, z_low=-5, z_high=5, color_obs=color_obstacle)
ax.legend()
#set equal axis
# ax.axis('equal')
fig.tight_layout()

if USE_OBSTACLE or USE_DIRECTIONAL_PEW_PEW_OBSTACLE or USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
    #plot the obstacles
    for i in range(N_obstacles):
        obstacle = {
            'x': [obs_x[i]],
            'y': [obs_y[i]],
            'z': [0],
            'radii': [random_radii[i]]
        }
        data_vis.plot_obstacles_3D(obstacle, ax, z_low=-5, z_high=5, color_obs=color_obstacle)
    
### animate the solution 

fig,ax, animate = data_vis.animate_trajectory_3D(entire_solution, use_time_span=False, time_span=10,
                                                 animation_interval=20, show_velocity=True,
                                                 vel_min=control_constraints['v_cmd_min'],
                                                 vel_max=control_constraints['v_cmd_max'])
ax.scatter(final_states[0], final_states[1], final_states[2], color=goal_color, s=100,
              label='Goal')  # Point in red
data_vis.plot_obstacles_3D(goal_params, ax, z_low=-5, z_high=5, color_obs=goal_color)

if USE_OBSTACLE or USE_DIRECTIONAL_PEW_PEW_OBSTACLE or USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
    #plot the obstacles
    for i in range(N_obstacles):
        obstacle = {
            'x': [obs_x[i]],
            'y': [obs_y[i]],
            'z': [0],
            'radii': [random_radii[i]]
        }
        data_vis.plot_obstacles_3D(obstacle, ax, z_low=-5, z_high=5, color_obs=color_obstacle)
ax.legend()
ax.set_title(title_video)
ax.view_init(39, -115, 0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#tight layout
fig.tight_layout()


folder_dir = 'videos/mp4/'
import matplotlib.animation as animation 
writervideo = animation.FFMpegWriter(fps=60) 
if SAVE_GIF:
    animate.save(folder_dir+title_video+'.mp4', writer=writervideo) 
    #animate.save('directional_effector_obs.gif', writer='imagemagick', fps=60)

fig, ax = data_vis.plot_controls(entire_solution, time_span, plane.n_controls)

#solution times
fig,ax = plt.subplots()
ax.plot(solution_times)
ax.set_title('Solution Times')
ax.set_xlabel('Iteration')

plt.show()