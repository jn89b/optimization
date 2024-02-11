
import numpy as np
import casadi as ca

from matplotlib import pyplot as plt

from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneOptControl import PlaneOptControl

def plot_controls(solution:dict, time_list:np.ndarray, n_controls:int):
    #plot controls
    fig,ax = plt.subplots(nrows=n_controls, figsize=(10,10))
    ax[0].plot(time_list[:-1], solution['u_phi'], 'r')
    ax[1].plot(time_list[:-1], solution['u_theta'], 'g')
    ax[2].plot(time_list[:-1], solution['u_psi'], 'b')
    ax[3].plot(time_list[:-1], solution['v_cmd'], 'k')
    
    return fig,ax 

mpc_params = {
    'N': 30,
    'Q': ca.diag([0.1, 0.1, 0, 0, 0, 0]),
    'R': ca.diag([0, 0, 0, 0]),
    'dt': 0.1
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(15),
    'u_theta_max': np.deg2rad(15),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   3,
    'v_cmd_max':   6
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
                        0, #z
                        0, #phi
                        0, #theta
                        np.deg2rad(45), #psi# 3  #airspeed
                        ]) 

final_states = np.array([15, #x
                         15, #y
                         0, #z
                         0,  #phi
                         0,  #theta
                         0,  #psi
                         ]) 

init_controls = np.array([0, 
                          0, 
                          0, 
                          control_constraints['v_cmd_min']])



plane = Plane()
plane.set_state_space()

USE_BASIC = False
USE_OBS_AVOID = False
USE_DYNAMIC_THREATS = True
USE_PEW_PEW = False
plt.close('all')

#%% Use the basic MPC
if USE_BASIC:
    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane
    )
    plane_mpc.init_optimization_problem()
    solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)
    
#%% Use the obstacle avoidance MPC
elif USE_OBS_AVOID:
    N_obstacles = 5
    obs_x = np.random.uniform(3, 8, N_obstacles)
    obs_y = np.random.uniform(3, 8, N_obstacles)
    random_radii = np.random.uniform(0.5, 0.9, N_obstacles)

    obs_avoid_params = {
        'weight': 1E-6,
        'safe_distance': 0.5,
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
    solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)
    
    fig,ax = plt.subplots(1, figsize=(10,10))
    if obs_avoid_params is not None:
        obs_x = obs_avoid_params['x']
        obs_y = obs_avoid_params['y'] 
        obs_radii = obs_avoid_params['radii']
        for i in range(len(obs_x)):
            circle = plt.Circle((obs_x[i], obs_y[i]), obs_radii[i], color='b', fill=False)
            ax.add_artist(circle)

    ax.plot(solution_results['x'], solution_results['y'], 'r')
    plt.show()
    
#%% DYNAMIC THREATS
elif USE_DYNAMIC_THREATS:
    num_dynamic_threats = 1
    threat_weight = 1E-1
    dynamic_threats = []
    threat_x_positions = [5] #[2, 0]
    threat_y_positions = [0] #[0, 5]
    # final_position = np.array([10, 10, np.deg2rad(225)])

    #feed the final position as my projection of where I will be at the end of the trajectory
    los_to_goal = np.arctan2(final_states[1] - init_states[1], final_states[0] - init_states[0])
    # projected_final_position = np.array([init_states[0] + 5*np.cos(los_to_goal),
    #                                     init_states[1] + 5*np.sin(los_to_goal),
    #                                     los_to_goal])
    projected_init_state = np.copy(init_states)
    for i in range(mpc_params['N']):
        final_position = np.array([projected_init_state[0] + mpc_params['dt']*np.cos(los_to_goal),
                                            projected_init_state[1] + mpc_params['dt']*np.sin(los_to_goal),
                                            los_to_goal])
        
        projected_init_state = np.copy(final_position)
        print('projected_final_position', final_position)
    
    final_position = np.array([10, 10, 0])
 
    for i in range(num_dynamic_threats):
        # threat_x = np.random.uniform(-3, 3)
        # threat_y = np.random.uniform(1.5, 4)
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
    
    dynamic_threat_params = {
        'threats':dynamic_threats, 
        'num_dynamic_threats':num_dynamic_threats,
        'weight': threat_weight,
        'time_index' : 5
    }    

    plane_mpc = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane,
        use_dynamic_threats=True,
        dynamic_threat_params=dynamic_threat_params
    )
    
    plane_mpc.init_optimization_problem()
    solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)
    time_vec = np.linspace(0, mpc_params['N']*mpc_params['dt'], mpc_params['N']+1)

    fig,ax = plt.subplots(1, figsize=(10,10))
    #plot as time gradient
    time_color = np.linspace(0, 1, mpc_params['N']+1)
    ax.scatter(solution_results['x'], solution_results['y'], c=time_color, cmap='viridis', marker='x', 
            label='Plane trajectory')
    
    for i, threat in enumerate(dynamic_threats):
        ax.scatter(threat.x_traj, threat.y_traj, c=time_color, cmap='viridis', marker='o', 
                label='Threat trajectory'+str(i))
        
    #show color bar
    cbar = plt.colorbar(ax.scatter(solution_results['x'], solution_results['y'], c=time_color, cmap='viridis', marker='x'))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    #plot controls
    fig,ax = plt.subplots(nrows=plane.n_controls, figsize=(10,10))
    ax[0].plot(time_vec[:-1], solution_results['u_phi'], 'r')
    ax[1].plot(time_vec[:-1], solution_results['u_theta'], 'g')
    ax[2].plot(time_vec[:-1], solution_results['u_psi'], 'b')
    ax[3].plot(time_vec[:-1], solution_results['v_cmd'], 'k')
    
    
    distance_from_threat = np.sqrt((solution_results['x'] - threat.x_traj)**2 + (solution_results['y'] - threat.y_traj)**2)
    fig,ax = plt.subplots(1, figsize=(10,10))
    ax.plot(time_vec[:], distance_from_threat, linestyle='-', marker='o', color='r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance from Threat')
    
    #change the position of the threats and recompute the trajectory    
    dynamic_threats = []
    num_dynamic_threats = 2
    for i in range(num_dynamic_threats):
        threat_x = np.random.uniform(-10, -1)
        threat_y = np.random.uniform(-10, -1)
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
    
    
    dynamic_threat_params = {
        'threats':dynamic_threats, 
        'num_dynamic_threats':num_dynamic_threats,
        'weight': threat_weight,
        'time_index': 5 #check the other time indicies 
    }    
    
    # plane_mpc.update_dynamic_threats(dynamic_threat_params)
    # print("dynamic threats updated", plane_mpc.dynamic_threat_params)
    # plane_mpc.init_optimization_problem()
    # solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)    

    # fig,ax = plt.subplots(1, figsize=(10,10))
    # #plot as time gradient
    # time_color = np.linspace(0, 1, mpc_params['N']+1)
    # ax.scatter(solution_results['x'], solution_results['y'], c=time_color, cmap='viridis', marker='x', 
    #         label='Plane trajectory')
    
    # for i, threat in enumerate(dynamic_threats):
    #     ax.scatter(threat.x_traj, threat.y_traj, c=time_color, cmap='viridis', marker='o', 
    #             label='Threat trajectory'+str(i))
        
    # #show color bar
    # cbar = plt.colorbar(ax.scatter(solution_results['x'], solution_results['y'], c=time_color, cmap='viridis', marker='x'))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.legend()
    plt.show()
#%% PEW PEW
elif USE_PEW_PEW:
        
    effector_config = {
            'effector_range': 10, 
            'effector_power': 1, 
            'effector_type': 'directional_3d', 
            'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
            'weight': 0.0, 
            'radius_target': 0.5
            }
    
    obs_avoid_params = {
        'weight': 1E-6,
        'safe_distance': 0.5,
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
    
    plane_mpc.init_optimization_problem()
    solution_results = plane_mpc.get_solution(init_states, final_states, init_controls)
    
    fig,ax = plt.subplots(1, figsize=(10,10))
    ax.plot(solution_results['x'], solution_results['y'], 'r')
    #plot the goal position
    ax.scatter(final_states[0], final_states[1], marker='x', color='b')

    #plot the controls
    time_vec = np.linspace(0, mpc_params['N']*mpc_params['dt'], mpc_params['N']+1)
    plot_controls(solution_results, time_vec, plane.n_controls)
        
    plt.show()
