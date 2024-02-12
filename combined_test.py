import numpy as np
import casadi as ca

from matplotlib import pyplot as plt
from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneOptControl import PlaneOptControl

"""
Testing the coupling for the optimization process for :
- Obstacle avoidance with effector
- Obstacle avoidance with dynamic threats 

Idea is to see if I can have a decision machine make the decision to avoid obstacles and threats
"""


mpc_params = {
    'N': 20,
    'Q': ca.diag([1E-3, 1E-3, 0, 0, 0, 0]),
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

final_states = np.array([10, #x
                         10, #y
                         0, #z
                         0,  #phi
                         0,  #theta
                         0,  #psi
                         ]) 

init_controls = np.array([0, 
                          0, 
                          0, 
                          control_constraints['v_cmd_max']])

plane = Plane()
plane.set_state_space()

seed = 0
np.random.seed(seed)

#%% Effector with obstacles
effector_config = {
        'effector_range': 10, 
        'effector_power': 1, 
        'effector_type': 'directional_3d', 
        'effector_angle': np.deg2rad(60), #double the angle of the cone, this will be divided to two
        'weight': 1E-3, 
        'radius_target': 0.5
        }

N_obstacles = 2
obs_x = np.random.uniform(4, 6, N_obstacles)
obs_y = np.random.uniform(5, 5, N_obstacles)
random_radii = np.random.uniform(0.5, 0.9, N_obstacles)

obs_avoid_params = {
    'weight': 1E-3,
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
    use_pew_pew=True,
    pew_pew_params=effector_config,
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

