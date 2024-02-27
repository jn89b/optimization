import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import time 
from controls.PlaneModel2 import PlaneModel2
from controls.PlaneOptControl2 import PlaneOptControl2 
from data_vis.DataVisualizer import DataVisualizer
# from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
###### INITIAL CONFIGURATIONS ########

plane = PlaneModel2()
plane.set_state_space()
data_vis = DataVisualizer()

#### PLOT CONFIGURATIONS ####
color_obstacle = 'purple'
trajectory_color = 'blue'
goal_color = 'green'

#### SET YOUR CONFIGURATIONS HERE #######
seed = 0
np.random.seed(seed)
sim_iteration = 100
idx_next_step = 2 #index of the next step in the solution
N_obstacles = 10


title_video = 'Omni Directional Effector Obstacle Avoidance'
SAVE_GIF = False

OBX_MIN_RANGE = 30
OBX_MAX_RANGE = 200
OBX_MIN_RADIUS = 10
OBX_MAX_RADIUS = 20

get_cost = True
 
USE_BASIC = False
USE_OBSTACLE = True
USE_TIME_CONSTRAINT = False
USE_DIRECTIONAL_PEW_PEW = False
USE_DIRECTIONAL_PEW_PEW_OBSTACLE = False
USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE = False


effector_config = {
        'effector_range': 200, 
        'effector_power': 1, 
        'effector_type': 'directional_3d', 
        'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
        'weight': 10, 
        'radius_target': 0.5
        }

mpc_params = {
    'N': 30,
    'Q': ca.diag([1E-2, 1E-2, 1E-2, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0.1, 0.1, 0.1, 0.1]),
    'dt': 0.1
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'load_z_min': -3,
    'load_z_max':  6,
    'load_x_min':  -2,
    'load_x_max':   2,
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
    'airspeed_min': 15,
    'airspeed_max': 30
}

init_controls = np.array([0, #load_x
                          0, #load_z
                          0, #u_phi
                          20])#v_cmd not really used in the model

init_states = np.array([0, #x
                        0, #y
                        0, #z
                        0, #phi
                        0, #theta
                        0, #psi
                        20])#v


final_states = np.array([300, #x
                            300, #y
                            5, #z
                            0, #phi
                            0, #theta
                            0, #psi
                            20])#v

goal_params = {
    'x': [final_states[0]],
    'y': [final_states[1]],
    'z': [final_states[2]],
    'radii': [1.0]
}   

# Create the model
plane = PlaneModel2()
plane.set_state_space()

if USE_BASIC:
    plane_mpc = PlaneOptControl2(control_constraints=control_constraints,
                                state_constraints=state_constraints,
                                mpc_params=mpc_params,
                                casadi_model=plane)

if USE_OBSTACLE:
    obs_x = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_obstacles)
    obs_y = np.random.randint(OBX_MIN_RANGE, OBX_MAX_RANGE, N_obstacles)
    random_radii = np.random.randint(OBX_MIN_RADIUS, OBX_MAX_RADIUS, N_obstacles)
    obs_avoid_params = {
        'weight': 1,
        'safe_distance': 3,
        'x': obs_x,
        'y': obs_y,
        'radii': random_radii,

    }
    plane_mpc = PlaneOptControl2(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane,
        use_obstacle_avoidance=True,
        obs_params=obs_avoid_params
    )

plane_mpc.init_optimization_problem()

solution_history = []
solution_times = []
driveby_locations = []
for i in range(sim_iteration):
    
    if USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
        driveby_direction = find_driveby_direction(final_states[:2], init_states[:2], 
                                                init_states[5], 
                                                effector_config['effector_range'],
                                                effector_config['minor_radius'],
                                                obs_avoid_params['radii'][-1],
                                                obs_avoid_params['safe_distance'])
        driveby_states = final_states.copy()        
        driveby_states[0] = driveby_direction[0]
        driveby_states[1] = driveby_direction[1]
        
        driveby_states[2] = 28
        driveby_states[3] = np.deg2rad(32.5)
        
        print("Drive By Direction: ", driveby_direction)
        start_time = time.time()
        solution_results,end_time = plane_mpc.get_solution(init_states, driveby_states, init_controls,
                                                  get_cost=get_cost)
        driveby_locations.append(driveby_direction)
        
        
    else:
        start_time = time.time()
        solution_results,end_time = plane_mpc.get_solution(init_states, final_states, init_controls,
                                                  get_cost=get_cost)
    final_time = end_time - start_time
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
    next_load_x = solution_results['load_x'][idx_next_step]
    next_load_z = solution_results['load_z'][idx_next_step]
    next_u_phi = solution_results['u_phi'][idx_next_step]
    next_v_cmd = solution_results['v_cmd'][idx_next_step]
    init_controls = np.array([next_load_x, next_load_z, next_u_phi, next_v_cmd])
    
    print("init_states: ", init_states)
    print("init_controls: ", init_controls)
    
    #prune out the solution to the next step
    if get_cost:
        solution_results = {
            'x': solution_results['x'][:idx_next_step],
            'y': solution_results['y'][:idx_next_step],
            'z': solution_results['z'][:idx_next_step],
            'phi': solution_results['phi'][:idx_next_step],
            'theta': solution_results['theta'][:idx_next_step],
            'psi': solution_results['psi'][:idx_next_step],
            'v': solution_results['v'][:idx_next_step],
            'load_x': solution_results['load_x'][:idx_next_step],
            'load_z': solution_results['load_z'][:idx_next_step],
            'u_phi': solution_results['u_phi'][:idx_next_step],
            'v_cmd': solution_results['v_cmd'][:idx_next_step],
            'cost': solution_results['cost'],
            'grad': solution_results['grad']
        }
    else:
        solution_results = {
            'x': solution_results['x'][:idx_next_step],
            'y': solution_results['y'][:idx_next_step],
            'z': solution_results['z'][:idx_next_step],
            'phi': solution_results['phi'][:idx_next_step],
            'theta': solution_results['theta'][:idx_next_step],
            'psi': solution_results['psi'][:idx_next_step],
            'v': solution_results['v'][:idx_next_step],
            'load_x': solution_results['load_x'][:idx_next_step],
            'load_z': solution_results['load_z'][:idx_next_step],
            'u_phi': solution_results['u_phi'][:idx_next_step],
            'v_cmd': solution_results['v_cmd'][:idx_next_step]
        }
    
    solution_history.append(solution_results)

#%% 
entire_solution = data_vis.unpack_solution_list(solution_history)
#time vector based on length of the solution
time_span = []
for i in range(len(entire_solution['x'])):
    time_span.append(i * mpc_params['dt'])

### Plot the solutionax.sc
fig,ax = data_vis.plot_trajectory_3d(entire_solution, use_time_color=True, time_list=time_span)
#plot the goal location
ax.scatter(final_states[0], final_states[1], final_states[2], color=goal_color, s=100, 
           label='Goal')  # Point in red
data_vis.plot_obstacles_3D(goal_params, ax, z_low=-5, z_high=5, color_obs=color_obstacle)
ax.legend()
#set equal axis
# ax.axis('equal')
fig.tight_layout()

if USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
    for i in range(len(driveby_locations)):
        ax.scatter(driveby_locations[i][0], driveby_locations[i][1], final_states[2], color='red', s=100, 
           label='Drive By')
        
    

if USE_OBSTACLE or USE_DIRECTIONAL_PEW_PEW_OBSTACLE or USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
    #plot the obstacles
    for i in range(N_obstacles):
        obstacle = {
            'x': [obs_x[i]],
            'y': [obs_y[i]],
            'z': [0],
            'radii': [random_radii[i]]
        }
        data_vis.plot_obstacles_3D(
            obstacle, ax, z_low=-5, z_high=5, color_obs=color_obstacle)
    
### animate the solution 

fig,ax, animate = data_vis.animate_trajectory_3D(
    entire_solution, use_time_span=False, time_span=10,
    animation_interval=20, show_velocity=True,
    vel_min=control_constraints['v_cmd_min'],
    vel_max=control_constraints['v_cmd_max'])

ax.scatter(final_states[0], final_states[1], 
           final_states[2], color=goal_color, s=100,
              label='Goal')  # Point in red
data_vis.plot_obstacles_3D(goal_params, ax, 
                           z_low=-5, z_high=5, color_obs=goal_color)

if USE_OBSTACLE or USE_DIRECTIONAL_PEW_PEW_OBSTACLE or \
    USE_OMNIDIRECTIONAL_PEW_PEW_OBSTACLE:
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

if get_cost:
    fig,ax = plt.subplots()
    ax.plot(entire_solution['cost'], marker='o')
    ax.set_title('Cost')
    ax.set_xlabel('Iteration')

    #fig,ax = plt.subplots()
    #ax.plot(entire_solution['grad'], marker='o')
    #ax.set_title('Gradient')
    #ax.set_xlabel('Iteration')
    
folder_dir = 'videos/mp4/'
import matplotlib.animation as animation 
writervideo = animation.FFMpegWriter(fps=60) 
if SAVE_GIF:
    animate.save(folder_dir+title_video+'.mp4', writer=writervideo) 
    #animate.save('directional_effector_obs.gif', writer='imagemagick', fps=60)


##compute the distasnce from goal for each iteration
distance_from_goal = []
for i in range(len(entire_solution['x'])):
    distance = np.linalg.norm([entire_solution['x'][i] - final_states[0],
                               entire_solution['y'][i] - final_states[1],
                               entire_solution['z'][i] - final_states[2]])
    distance_from_goal.append(distance)
    
fig, ax = data_vis.plot_controls_load(entire_solution, time_span, plane.n_controls,
                                 add_one_more=True, additional_row=distance_from_goal,
                                 label_name='Distance from Goal')
#SHARE ax
ax.sharex(ax)

fig,ax = data_vis.plot_attitudes(entire_solution, time_span, plane.n_states)


#solution times
fig,ax = plt.subplots()
ax.plot(solution_times, marker='o')
ax.set_title('Solution Times')
ax.set_xlabel('Iteration')

plt.show()



