import casadi as ca


import numpy as np
import casadi as ca
import time 
from matplotlib import pyplot as plt
from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneOptControl import PlaneOptControl
from data_vis.DataVisualizer import DataVisualizer


plane = Plane(compile_to_c=True, use_compiled_fn=True)
plane.set_state_space()
print(plane.function)

plane_no_compiled = Plane(compile_to_c=False, use_compiled_fn=False)
plane_no_compiled.set_state_space()

#test rk45
N_steps = 1000
x = np.array([0, 0, 0, 0, 0, 0, 0])
u = np.array([0, 0, 0, 0])
dt = 0.05

t_compiled_history = []
for i in range(N_steps):
    t_init = time.time()
    plane.rk45(x, u, dt)
    t_final = time.time()
    t_compiled_history.append(t_final - t_init)

t_no_compiled = []
for i in range(N_steps):
    t_init = time.time()
    plane_no_compiled.rk45(x, u, dt)
    t_final = time.time()
    t_no_compiled.append(t_final - t_init)
    
print("mean time compiled: ", np.mean(t_compiled_history))
print("mean time no compiled: ", np.mean(t_no_compiled))


##### Test Compiled MPC #####
mpc_params = {
    'N': 10,
    'Q': ca.diag([1.0, 1.0, 1.0, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0.1, 0.1, 0.1, 0.1]),
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
                        1, #y
                        0, #z
                        0, #phi
                        0, #theta
                        np.deg2rad(0), #psi# 3  \
                        15 #airspeed
                        ]) 

final_states = np.array([100, #x
                         250, #y
                         0, #zod
                         0,  #phi
                         0,  #theta
                         0,  #psi
                         15 #airspeed
                         ]) 

init_controls = np.array([0, 
                          0, 
                          0, 
                          control_constraints['v_cmd_min']])

goal_params = {
    'x': [final_states[0]],
    'y': [final_states[1]],
    'z': [final_states[2]],
    'radii': [1.0]
}   

N_obstacles = 3
OBX_MIN_RANGE = 30
OBX_MAX_RANGE = 200
OBX_MIN_RADIUS = 32
OBX_MAX_RADIUS = 45

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


obs_avoid_params = {
    'weight': 0.1,
    'safe_distance': 3.0,
    'x': obs_x,
    'y': obs_y,
    'radii': random_radii
}

effector_config = {
        'effector_range': 100, 
        'effector_power': 1, 
        'effector_type': 'directional_3d', 
        'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
        'weight': 100, 
        'radius_target': 2.0
        }

plane_mpc_compiled_everything = PlaneOptControl(
    control_constraints, 
    state_constraints, 
    mpc_params, 
    plane_no_compiled,
    use_pew_pew=True,
    pew_pew_params=effector_config,
    use_obstacle_avoidance=True,
    obs_params=obs_avoid_params,
    compile_to_c=False,
    use_c_code=True
)
    
plane_mpc_compiled_everything.init_optimization_problem()

num_sim_steps = 100

compiled_time_history = []
idx_next_step = 2
for i in range(num_sim_steps):
    start_time = time.time()
    solution_results,end_time = plane_mpc_compiled_everything.get_solution(
        init_states,
        final_states,
        init_controls
    )
    soln_end_time = time.time()
    compiled_time_history.append(soln_end_time - start_time)

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
    
    
print("mpc time compiled: ", np.mean(compiled_time_history))
print("standard deviation: ", np.std(compiled_time_history))
print("max time: ", np.max(compiled_time_history))
print("final states: ", init_states)


#### Not compiled ####
plane_mpc_not_compiled = PlaneOptControl(
    control_constraints, 
    state_constraints, 
    mpc_params, 
    plane_no_compiled,
    use_pew_pew=True,
    pew_pew_params=effector_config,
    use_obstacle_avoidance=True,
    obs_params=obs_avoid_params,
)
    
plane_mpc_not_compiled.init_optimization_problem()

num_sim_steps = 100

init_states = np.array([0, #x 
                        1, #y
                        0, #z
                        0, #phi
                        0, #theta
                        np.deg2rad(0), #psi# 3  \
                        15 #airspeed
                        ]) 

final_states = np.array([100, #x
                         250, #y
                         0, #zod
                         0,  #phi
                         0,  #theta
                         0,  #psi
                         15 #airspeed
                         ]) 

init_controls = np.array([0, 
                          0, 
                          0, 
                          control_constraints['v_cmd_min']])

not_compiled_history = []
for i in range(num_sim_steps):
    start_time = time.time()
    solution_results,end_time = plane_mpc_not_compiled.get_solution(
        init_states,
        final_states,
        init_controls
    )
    soln_end_time = time.time()
    not_compiled_history.append(soln_end_time - start_time)

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
    

print("mpc time not compiled: ", np.mean(not_compiled_history))
print("standard deviation: ", np.std(not_compiled_history))
print("final states: ", init_states)
print("obstacles: ", obs_x, obs_y, random_radii)

#plot the time results
plt.figure()
plt.plot(compiled_time_history, label='compiled')
plt.plot(not_compiled_history, label='not compiled')

plt.xlabel('Time Step')
plt.ylabel('Time (s)')

plt.legend()
plt.show()