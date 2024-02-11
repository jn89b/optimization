import casadi as ca
import numpy as np

from matplotlib import pyplot as plt
from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneOptControl import PlaneOptControl
import multiprocess as mp
import time as time 

# from pathos.multiprocessing import ProcessingPool as Pool

# x = MX.sym("x")


# solver = nlpsol("solver","ipopt",{"x":x,"f":sin(x)**2})
# solver2 = nlpsol("solver2","ipopt",{"x":x,"f":cos(x)**2})


# def mysolve(x0):
#   return solver(x0=x0)["x"]

# p = Pool(3)

# print(p.map(mysolve, [0.1, 0.2, 0.3]))


# def solve_problem1():
#     x = ca.SX.sym('x')
#     f = ca.sin(x)**2
#     solver = ca.nlpsol("solver", "ipopt", {"x":x, "f":f})
#     solution = solver(x0=0.5) # Example initial guess
#     return solution["x"]

# def solve_problem2():
#     x = ca.SX.sym('x')
#     f = ca.cos(x)**2
#     solver2 = ca.nlpsol("solver2", "ipopt", {"x":x, "f":f})
#     solution = solver2(x0=0.5) # Example initial guess
#     return solution["x"]

# if __name__ == "__main__":
#     with ProcessPoolExecutor() as executor:
#         future1 = executor.submit(solve_problem1)
#         future2 = executor.submit(solve_problem2)
#         result1 = future1.result()
#         result2 = future2.result()
#         print("Result of solver 1:", result1)
#         print("Result of solver 2:", result2)
        

def solve_basic_optimization(init_states:np.ndarray, 
                      final_states:np.ndarray, 
                      init_controls:np.ndarray,
                      queue) -> dict:
    """
    Find the optimal control for the plane
    """
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
    basic_opt = PlaneOptControl(
        control_constraints, 
        state_constraints, 
        mpc_params, 
        plane
    )

    basic_opt.init_optimization_problem()
    solution_results = basic_opt.get_solution(
        init_states, final_states, init_controls)

    queue.put(solution_results)
    print("SOLVED THE BASIC")

    #return solution_results

def test_basic(init_states:np.ndarray,
                            final_states:np.ndarray, 
                            init_controls:np.ndarray,
                            obstacle_opti:PlaneOptControl,
                            queue) -> dict:
    start_time = time.time()
    
    solution_results = obstacle_opti.get_solution(
        init_states, final_states, init_controls)
    
    queue.put(solution_results)
    print("SOLVED THE BASIC took", time.time() - start_time, "seconds")

def solve_obstacle_avoidance(init_states:np.ndarray,
                            final_states:np.ndarray, 
                            init_controls:np.ndarray,
                            obstacle_opti:PlaneOptControl,
                            queue) -> dict:
    """
    Find the optimal control for the plane
    """
    start_time = time.time()
    #can initialize this outside of the function
    #obstacle_opti.init_optimization_problem()

    solution_results = obstacle_opti.get_solution(
        init_states, final_states, init_controls)
    # print("SOLVED THE OBSTACLE AVOIDANCE", solution_results)
    queue.put(solution_results)
    print("SOLVED THE AVOIDANCE took", time.time() - start_time, "seconds")

    # return solution_results

def solve_dynamic_threats(init_states:np.ndarray, 
                          final_states:np.ndarray, 
                          init_controls:np.ndarray,
                          dynamic_opti:PlaneOptControl) -> dict:
    """
    Find the optimal control for the plane
    """
    dynamic_opti.init_optimization_problem()
    solution_results = dynamic_opti.get_solution(
        init_states, final_states, init_controls)

    return solution_results


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


#%% --------------- BASIC OPTIMIZATION ------------------------------------
basic_opt = PlaneOptControl(
    control_constraints, 
    state_constraints, 
    mpc_params, 
    plane
)

basic_opt.init_optimization_problem()

# #%% Obstacle avoidance 
N_obstacles = 5
obs_x = np.random.uniform(4, 8, N_obstacles)
obs_y = np.random.uniform(4, 8, N_obstacles)
random_radii = np.random.uniform(0.5, 0.9, N_obstacles)

obs_avoid_params = {
    'weight': 1E-3,
    'safe_distance': 0.5,
    'x': obs_x,
    'y': obs_y,
    'radii': random_radii
}

obstacle_opt = PlaneOptControl(
    control_constraints, 
    state_constraints, 
    mpc_params, 
    plane,
    use_obstacle_avoidance=True,
    obs_params=obs_avoid_params
)
obstacle_opt.init_optimization_problem()

# #%% ---------------DYNAMIC THREAT OPTIMIZATION ------------------------------------
# num_dynamic_threats = 1
# threat_weight = 1E-1
# dynamic_threats = []
# threat_x_positions = [5] #[2, 0]
# threat_y_positions = [0] #[0, 5]
# # final_position = np.array([10, 10, np.deg2rad(225)])

# #feed the final position as my projection of where I will be at the end of the trajectory
# los_to_goal = np.arctan2(final_states[1] - init_states[1], final_states[0] - init_states[0])
# # projected_final_position = np.array([init_states[0] + 5*np.cos(los_to_goal),
# #                                     init_states[1] + 5*np.sin(los_to_goal),
# #                                     los_to_goal])
# projected_init_state = np.copy(init_states)
# for i in range(mpc_params['N']):
#     final_position = np.array([projected_init_state[0] + mpc_params['dt']*np.cos(los_to_goal),
#                                         projected_init_state[1] + mpc_params['dt']*np.sin(los_to_goal),
#                                         los_to_goal])
    
#     projected_init_state = np.copy(final_position)

# print('projected_final_position', final_position)

# final_position = np.array([10, 10, 0])

# for i in range(num_dynamic_threats):
#     # threat_x = np.random.uniform(-3, 3)
#     # threat_y = np.random.uniform(1.5, 4)
#     threat_x = threat_x_positions[i]
#     threat_y = threat_y_positions[i]
#     threat_psi = np.arctan2(final_position[1] - threat_y,
#                             final_position[0] - threat_x)
#     threat_position = np.array([threat_x, threat_y, threat_psi])
    
#     threat_params = {
#         'safe_distance': 0.5,
#         'velocity': 5,
#         'type': 'holonomic'
#     }
    
#     algo_params = {
#         'final_position':final_position, 
#         'num_points':mpc_params['N']
#     }
    
#     threat = Threat(init_position=threat_position,
#                     algos_params={'final_position':final_position, 
#                                 'num_points':mpc_params['N']},
#                     threat_params=threat_params,
#                     use_2D=True)
    
#     if threat.use_2D:
#         threat.straight_line_2D(final_position, mpc_params['N'], mpc_params['dt'])
        
#     if threat.use_3D:
#         threat.straight_line_3D(final_position, mpc_params['N'], mpc_params['dt'])
    
#     dynamic_threats.append(threat)

# dynamic_threat_params = {
#     'threats':dynamic_threats, 
#     'num_dynamic_threats':num_dynamic_threats,
#     'weight': threat_weight,
#     'time_index' : 5
# }    
# dynamic_opt = PlaneOptControl(
#     control_constraints, 
#     state_constraints, 
#     mpc_params, 
#     plane,
#     use_dynamic_threats=True,
#     dynamic_threat_params=dynamic_threat_params
# )
    
#%% Run parallel processes
# Run Concurrent processes
num_workers = 2

# with ProcessPoolExecutor() as executor:
#     future1 = executor.submit(solve_basic_optimization, 
#                               init_states, 
#                               final_states, 
#                               init_controls)
#     # future2 = executor.submit(solve_obstacle_avoidance, 
#     #                           init_states, 
#     #                           final_states, 
#     #                           init_controls,
#     #                           obstacle_opt)
#     # future3 = executor.submit(solve_dynamic_threats, 
#     #                           init_states, 
#     #                           final_states, 
#     #                           init_controls,
#     #                           dynamic_opt)
#     result1 = future1.result()
#     # result2 = future2.result()
#     # result3 = future3.result()
#     print("Result of basic optimization:", result1)
    # print("Result of obstacle avoidance:", result2)
    # print("Result of dynamic threats:", result3)
    
    
#use pathos
# with Pool(num_workers) as executor:
#     future1 = executor.apipe(solve_basic_optimization, 
#                             init_states, 
#                             final_states, 
#                             init_controls)
#     future2 = executor.apipe(solve_obstacle_avoidance, 
#                             init_states, 
#                             final_states, 
#                             init_controls,
#                             obstacle_opt)
#     # future3 = executor.map(solve_dynamic_threats, 
#     #                         init_states, 
#     #                         final_states, 
#     #                         init_controls,
#     #                         dynamic_opt)
#     result1 = future1.get()
#     result2 = future2.get()
#     # result3 = future3
#     print("Result of basic optimization:", result1)
#     print("Result of obstacle avoidance:", result2)
    # print("Result of dynamic threats:", result3)
    
# use multiprocess
queue = mp.Queue()

# p1 = mp.Process(target=test_basic,
#                 args=(init_states, final_states, init_controls, basic_opt, queue))\
p1 = mp.Process(target=solve_basic_optimization,
                args=(init_states, final_states, init_controls, queue))

# p2 = mp.Process(target=solve_obstacle_avoidance,
#                 args=(init_states, final_states, init_controls, obstacle_opt, queue))

start_time = time.time()
#start the process
p1.start()
# p2.start()

#wait for process to finish
p1.join()
# p2.join()

results = [queue.get() for _ in range(1)]

#close the queue
queue.close()

print("final time taken", time.time() - start_time, "seconds")
