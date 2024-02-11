### Formulate the MPC problem with obstacles
import numpy as np
import casadi as ca

from matplotlib import pyplot as plt
from controls.OptiMPC import OptiCasadi
from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneMPC import PlaneMPC


def format_mpc_trajectory(sol:ca.OptiSol, 
                          state_idx_dict:dict, 
                          control_idx_dict:dict) -> tuple:
    states   = sol.value(mpc_controller.X)
    controls = sol.value(mpc_controller.U)
    
    state_results = {}
    for key, value in state_idx_dict.items():
        if value >= states.shape[0]:
            continue
        
        state_results[key] = states[value, :]
        
    control_results = {}
    for key, value in control_idx_dict.items():
        control_results[key] = controls[value, :]
        
    return state_results, control_results

mpc_params = {
    'N': 20,
    'Q': np.diag([1, 1, 1, 1, 1, 1, 1]),
    'R': np.diag([1, 1, 1, 1]),
    'dt': 0.1
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(15),
    'u_theta_max': np.deg2rad(15),
    'u_psi_min':  -np.deg2rad(30),
    'u_psi_max':   np.deg2rad(30),
    'v_cmd_min':   3,
    'v_cmd_max':   6
}

state_constraints = {
    'x_dot_min': -np.inf,
    'x_dot_max': np.inf,
    'y_dot_min': -np.inf,
    'y_dot_max': np.inf,
    'z_dot_min': -10,
    'z_dot_max': 50,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(15),
    'theta_max': np.deg2rad(15),
    'psi_min':  -np.deg2rad(180),
    'psi_max':   np.deg2rad(180),
    'airspeed_min': 1,
    'airspeed_max': 10
}

control_indices = {
    'u_phi': 0,
    'u_theta': 1,
    'u_psi': 2,
    'v_cmd': 3
}

state_indices = {
    'x_dot': 0,
    'y_dot': 1,
    'z_dot': 2,
    'phi_dot': 3,
    'theta_dot': 4,
    'psi_dot': 5,
    'airspeed': 6
    
}

init_states = np.array([1, #x 
                        1, #y
                        2, #z
                        0, #phi
                        0, #theta
                        np.deg2rad(45), #psi# 3  #airspeed
                        ]) 

final_states = np.array([10, #x
                         10, #y
                         10, #z
                         0,  #phi
                         0,  #theta
                         0,  #psi
                        #  3   #airspeed
                         ]) 

init_controls = np.array([0, 
                          0, 
                          0, 
                          control_constraints['v_cmd_min']])


plane = Plane()
plane.set_state_space()


USE_OBS_AVOIDANCE  = False
USE_DYNAMIC_THREAT = True
USE_TIME_MINIMIZE  = False

#%% Obstacle avoidance
if USE_OBS_AVOIDANCE:
    N_obstacles = 1
    obs_x = np.random.uniform(1.5, 8, N_obstacles)
    obs_y = np.random.uniform(1.5, 8, N_obstacles)
    random_radii = np.random.uniform(0.3, 0.8, N_obstacles)

    obs_avoid_params = {
        'weight': 5,
        'safe_distance': 0.2,
        'x': obs_x,
        'y': obs_y,
        'radii': random_radii
    }

    mpc_controller = PlaneMPC(mpc_params, 
                            control_constraints, 
                            state_constraints,
                            control_indices,
                            state_indices,
                            plane,
                            use_obs_avoidance=True,
                            obs_avoid_params=obs_avoid_params)

    mpc_controller.set_init_controls(init_controls)
    mpc_controller.init_optimal_control_prob(init_states, final_states)

    import time
    init_time = time.time()
    solution     = mpc_controller.solve()
    time_solve = time.time() - init_time
    print("Time to solve: ", time_solve)
    state_traj   = solution.value(mpc_controller.X)
    control_traj = solution.value(mpc_controller.U)

    state_dict, control_dict = format_mpc_trajectory(solution,
                                                        state_indices,
                                                        control_indices)
    print("SOLVING AGAIN...")
    solution     = mpc_controller.solve()

    fig,ax = plt.subplots(1, figsize=(10,10))
    if obs_avoid_params is not None:
        obs_x = obs_avoid_params['x']
        obs_y = obs_avoid_params['y']
        obs_radii = obs_avoid_params['radii']
        for i in range(len(obs_x)):
            circle = plt.Circle((obs_x[i], obs_y[i]), obs_radii[i], color='b', fill=False)
            ax.add_artist(circle)

    ax.plot(state_dict['x_dot'], state_dict['y_dot'], 'r')
    
#%% Dynamic threat avoidance 
if USE_DYNAMIC_THREAT:
    print("------------Using dynamic threat avoidance------------")
    init_states = np.array([1, #x 
                            1, #y
                            0, #z
                            0, #phi
                            0, #theta
                            np.deg2rad(45), #psi# 
                            0, #time
                            ]) 

    final_states = np.array([10, #x
                            10, #y
                            0, #z
                            0,  #phi
                            0,  #theta
                            0,  #psi
                            0.01, #time
                            ]) 

    init_controls = np.array([0, 
                            0, 
                            0, 
                            control_constraints['v_cmd_min']])
    
    plane = Plane(include_time=True, dt_val=mpc_params['dt'])
    plane.set_state_space()
    
    num_dynamic_threats = 1 
    dynamic_threats = []
    threat_x_positions = [5]
    threat_y_positions = [0]
    final_position = np.array([10, 10, np.deg2rad(225)])
    for i in range(num_dynamic_threats):
        threat_x = np.random.uniform(-3, 3)
        threat_y = np.random.uniform(1.5, 4)
        threat_x = threat_x_positions[i]
        threat_y = threat_y_positions[i]
        threat_psi = np.arctan2(final_position[1] - threat_y,
                                final_position[0] - threat_x)
        threat_position = np.array([threat_x, threat_y, threat_psi])
        
        threat_params = {
            'safe_distance': 0.5,
            'velocity': 5,
            'type': 'holonomic'
        }
        
        algo_params = {
            'final_position':final_position, 
            'num_points':mpc_params['N']
        }
        
        threat = Threat(init_position=threat_position,
                        algos_params={'final_position':final_position, 
                                    'num_points':mpc_params['N']},
                        threat_params=threat_params,
                        use_2D=True)
        
        if threat.use_2D:
            threat.straight_line_2D(final_position, mpc_params['N'], mpc_params['dt'])
            
        if threat.use_3D:
            threat.straight_line_3D(final_position, mpc_params['N'], mpc_params['dt'])
        
        dynamic_threats.append(threat)

    # threat_position = np.array([3, 2, np.deg2rad(225)])
    
    # #compute los from start to end 
    # los = np.arctan2(final_position[1] - threat_position[1], 
    #                  final_position[0] - threat_position[0])
    
    # threat_position[2] = los

    # threat_params = {
    #     'safe_distance': 0.5,
    #     'velocity': 5,
    #     'type': 'holonomic'
    # }
    
    # algo_params = {
    #     'final_position':final_position, 
    #     'num_points':mpc_params['N']
    # }
    
    # threat = Threat(init_position=threat_position,
    #                 algos_params={'final_position':final_position, 
    #                               'num_points':mpc_params['N']},
    #                 threat_params=threat_params,
    #                 use_2D=True)
    
    # if threat.use_2D:
    #     threat.straight_line_2D(final_position, mpc_params['N'], mpc_params['dt'])
        
    # if threat.use_3D:
    #     threat.straight_line_3D(final_position, mpc_params['N'], mpc_params['dt'])

    #do one for now     
    
    dynamic_threat_params = {
        'threats':dynamic_threats, 
        'num_dynamic_threats':num_dynamic_threats,
        'weight': 5,
    }

    mpc_controller = PlaneMPC(mpc_params, 
                            control_constraints, 
                            state_constraints,
                            control_indices,
                            state_indices,
                            plane,
                            use_dynamic_threats=True,
                            dynamic_threat_params=dynamic_threat_params)
    
    mpc_controller.set_init_controls(init_controls)
    mpc_controller.init_optimal_control_prob(init_states, final_states)
    mpc_controller.set_solution_options()
    solution = mpc_controller.solve()
    
    state_dict, control_dict = format_mpc_trajectory(solution,
                                                        state_indices,
                                                        control_indices)

    time_vec = np.linspace(0, mpc_params['N']*mpc_params['dt'], mpc_params['N']+1)

    fig,ax = plt.subplots(1, figsize=(10,10))
    #plot as time gradient
    time_color = np.linspace(0, 1, mpc_params['N']+1)
    ax.scatter(state_dict['x_dot'], state_dict['y_dot'], c=time_color, 
               cmap='viridis', marker='x', 
               label='Plane trajectory')
    
    for i, threat in enumerate(dynamic_threats):
        ax.scatter(threat.x_traj, threat.y_traj, c=time_color, 
                   cmap='viridis', marker='o', 
                   label='Threat trajectory'+str(i))
        
    # ax.scatter(threat.x_traj, threat.y_traj, c=time_color, cmap='viridis', marker='o', 
    #            label='Threat trajectory')

    #show color bar
    cbar = plt.colorbar(ax.scatter(state_dict['x_dot'], 
                                   state_dict['y_dot'], 
                                   c=time_color, 
                                   cmap='viridis', marker='x'))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    #plot controls
    fig,ax = plt.subplots(nrows=plane.n_controls, figsize=(10,10))
    ax[0].plot(time_vec[:-1], control_dict['u_phi'], 'r')
    ax[1].plot(time_vec[:-1], control_dict['u_theta'], 'g')
    ax[2].plot(time_vec[:-1], control_dict['u_psi'], 'b')
    ax[3].plot(time_vec[:-1], control_dict['v_cmd'], 'k')
    
    
    distance_from_threat = np.sqrt((state_dict['x_dot'] - threat.x_traj)**2 + (state_dict['y_dot'] - threat.y_traj)**2)
    fig,ax = plt.subplots(1, figsize=(10,10))
    ax.plot(time_vec[:], distance_from_threat, linestyle='-', marker='o', color='r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance from Threat')
    

#%% Minimize time
if USE_TIME_MINIMIZE:
    
    init_states = np.array([1, #x 
                            1, #y
                            2, #z
                            0, #phi
                            0, #theta
                            np.deg2rad(45), #psi# 
                            0, #time
                            ]) 

    final_states = np.array([10, #x
                            10, #y
                            10, #z
                            0,  #phi
                            0,  #theta
                            0,  #psi
                            0.01, #time
                            ]) 

    init_controls = np.array([0, 
                            0, 
                            0, 
                            3])

    plane = Plane(include_time=True, dt_val=mpc_params['dt'])
    plane.set_state_space()

    mpc_controller = PlaneMPC(mpc_params, 
                            control_constraints, 
                            state_constraints,
                            control_indices,
                            state_indices,
                            plane,
                            use_minimize_time=True)

    mpc_controller.set_init_controls(init_controls)
    mpc_controller.init_optimal_control_prob(init_states, final_states)

    solution = mpc_controller.solve()
    state_traj   = solution.value(mpc_controller.X)
    control_traj   = solution.value(mpc_controller.U)

    state_dict, control_dict = format_mpc_trajectory(solution,
                                                        state_indices,
                                                        control_indices)

    time_vec = np.linspace(0, mpc_params['N']*mpc_params['dt'], mpc_params['N']+1)

    fig,ax = plt.subplots(1, figsize=(10,10))
    ax.plot(state_dict['x_dot'], state_dict['y_dot'], 'r')
    print("Final time is: ", solution.value(time_vec[-1]))

    fig,ax = plt.subplots(1, figsize=(10,10))
    ax.plot(time_vec[:], state_dict['x_dot'], 'r')
    ax.plot(time_vec[:], state_dict['y_dot'], 'g')

    #plot controls
    fig,ax = plt.subplots(nrows=plane.n_controls, figsize=(10,10))
    ax[0].plot(time_vec[:-1], control_dict['u_phi'], 'r')
    ax[1].plot(time_vec[:-1], control_dict['u_theta'], 'g')
    ax[2].plot(time_vec[:-1], control_dict['u_psi'], 'b')
    ax[3].plot(time_vec[:-1], control_dict['v_cmd'], 'k')

plt.show()