
import numpy as np
import casadi as ca

from matplotlib import pyplot as plt
from controls.OptiMPC import OptiCasadi
from models.Plane import Plane
from drone_control.Threat import Threat
from controls.PlaneMPC import PlaneMPC

from controls.OptimalControl import OptimalControlProblem

class PlaneOptControl(OptimalControlProblem):
    def __init__(self, 
                 control_constraints:dict, 
                 state_constraints:dict, 
                 mpc_params:dict, casadi_model,
                 use_obstacle_avoidance:bool=False,
                 obs_params:dict=None,
                 use_dynamic_threats:bool=False,
                 dynamic_threat_params:dict=None) -> None:
        
        super().__init__(mpc_params, casadi_model)
        self.control_constraints = control_constraints
        self.state_constraints = state_constraints
        # self.update_bound_constraints()
        self.cost = 0
        
        self.use_obstacle_avoidance = use_obstacle_avoidance
        self.obs_params = obs_params
        
        self.use_dynamic_threats = use_dynamic_threats
        self.dynamic_threat_params = dynamic_threat_params
        
    def update_bound_constraints(self) -> None:
        #add control constraints
        self.lbx['U'][0,:] = self.control_constraints['u_phi_min']
        self.ubx['U'][0,:] = self.control_constraints['u_phi_max']

        self.lbx['U'][1,:] = self.control_constraints['u_theta_min']
        self.ubx['U'][1,:] = self.control_constraints['u_theta_max']

        self.lbx['U'][2,:] = self.control_constraints['u_psi_min']
        self.ubx['U'][2,:] = self.control_constraints['u_psi_max']

        self.lbx['U'][3,:] = self.control_constraints['v_cmd_min']
        self.ubx['U'][3,:] = self.control_constraints['v_cmd_max']

        ### State Constraints
        self.lbx['X'][2,:] = self.state_constraints['z_min']
        self.ubx['X'][2,:] = self.state_constraints['z_max']

        self.lbx['X'][3,:] = self.state_constraints['phi_min']
        self.ubx['X'][3,:] = self.state_constraints['phi_max']

        self.lbx['X'][4,:] = self.state_constraints['theta_min']
        self.ubx['X'][4,:] = self.state_constraints['theta_max']
        print('Bound constraints updated')

    def compute_obstacle_avoidance_cost(self) -> ca.SX:
        obs_avoid_weight = self.obs_params['weight']
        obs_x_vector = self.obs_params['x']
        obs_y_vector = self.obs_params['y']
        obs_radii_vector = self.obs_params['radii']
        safe_distance = self.obs_params['safe_distance']
        
        x_position = self.X[0,:]
        y_position = self.X[1,:]
        
        avoidance_cost = 0
        
        for i,x in enumerate(obs_x_vector):
            obs_x = obs_x_vector[i]
            obs_y = obs_y_vector[i]
            obs_radii = obs_radii_vector[i]
            # obstacle_constraints = self.opti.variable(self.N+1)
            obstacle_distance = -ca.sqrt((obs_x - x_position)**2 + \
                (obs_y - y_position)**2)
            diff = obstacle_distance + obs_radii + safe_distance
            #obstacle_distance = 1/obstacle_distance
            avoidance_cost += obs_avoid_weight * ca.sum2(diff)
            
            self.g = ca.vertcat(self.g, diff[:-1].T)
        
        print('Obstacle avoidance cost computed')
        return avoidance_cost
    
    def compute_dynamic_threats_cost(self, cost:float) -> None:
        #ego vehicle
        print('Computing dynamic threats cost')
        x_pos    = self.X[0, :]
        y_pos    = self.X[1, :]
        z_pos    = self.X[2, :]
        psi      = self.X[5, :]        
        #this is a list of dynamic threats refer to the Threat class
        threat_list = self.dynamic_threat_params['threats'] 
        for threat in threat_list:
            threat_x_traj = threat.x_traj
            threat_y_traj = threat.y_traj
            # threat_z_traj = threat.z_traj
            threat_psi_traj = threat.psi_traj
            threat_time_traj = threat.time_traj
            #make sure that the threat trajectory is the same length as the ego vehicle
            if len(threat_x_traj) != self.N+1:
                raise ValueError('Threat trajectory must be the same length as the ego vehicle', 
                                 len(threat_x_traj), self.N+1)
                

            # distance = ca.sqrt((x_pos - threat_x_traj)**2 + \
            #     (y_pos - threat_y_traj)**2 + (z_pos - threat_z_traj)**2)

            distance = ca.sqrt((x_pos.T - threat_x_traj)**2 + \
                (y_pos.T - threat_y_traj)**2)

            threat_radii = 1.0 
            safe_distance = 1    
            #if the distance is less than the sum of the radii and the safe distance
            # then we are in danger
            difference = -distance + threat_radii + safe_distance
            
            distance = ca.sqrt((x_pos.T - threat_x_traj)**2 + \
                (y_pos.T - threat_y_traj)**2)

            #avoid division by zero
            distance = ca.if_else(distance < 1E-3, 1E-3, distance)
            distance_cost = 1/(distance)
            
            #compute the differnece of line of sight
            delta_psi = psi.T - threat_psi_traj
            
            #add the cost to the total cost
            threat_cost = self.dynamic_threat_params['weight'] * ca.sumsqr(distance_cost)
            cost += threat_cost
            
        return cost
    
    def compute_total_cost(self) -> ca.SX:
        self.cost += self.compute_dynamics_cost()
        if self.use_obstacle_avoidance:
            print('Using obstacle avoidance')
            self.cost += self.compute_obstacle_avoidance_cost()
            
        if self.use_dynamic_threats:
            print('Using dynamic threats')
            self.cost += self.compute_dynamic_threats_cost(self.cost)
            # self.cost +=self.cost
            
        return self.cost
    

    def solve(self, x0:np.ndarray, xF:np.ndarray, u0:np.ndarray) -> dict:
        """solve the optimal control problem"""
        
        state_init = ca.DM(x0)
        state_final = ca.DM(xF)
        
        X0 = ca.repmat(state_init, 1, self.N + 1)
        U0 = ca.repmat(u0, 1, self.N)

        n_states = self.model_casadi.n_states
        n_controls = self.model_casadi.n_controls
        
        if self.use_obstacle_avoidance:
            # constraints lower bound added 
            num_obstacles = len(self.obs_params['x'])
            num_constraints = num_obstacles * self.N
            lbg =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
            # -infinity to minimum marign value for obs avoidance  
            lbg[n_states*self.N+n_states:] = -ca.inf 
            
            # constraints upper bound
            ubg  =  ca.DM.zeros((n_states*(self.N+1)+num_constraints, 1))
            #rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
            ubg[n_states*self.N+n_states:] = 0
        else:
            lbg = ca.DM.zeros((n_states*(self.N+1), 1))
            ubg  =  ca.DM.zeros((n_states*(self.N+1), 1))
            
        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': self.pack_variables_fn(**self.lbx)['flat'],
            'ubx': self.pack_variables_fn(**self.ubx)['flat'],
        }
        
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_final   # target state
        )
        
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(self.N+1), 1),
            ca.reshape(U0, n_controls*self.N, 1)
        )

        sol = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        
        return sol
    
    def unpack_solution(self, sol:dict) -> np.ndarray:
        """
        This function unpacks the solution from the solver
        Original solution is a dictionary with keys 'x' and 'u' where 
        'x' is a matrix size of (n_states * (N+1), 1) and 
        'u' is a matrix size of (n_controls * N, 1) 
        
        This function reshapes the solution to the original state and control matrices
        where u is a matrix of size (n_controls, N) and x is a matrix of size (n_states, N+1)
        """
        u = ca.reshape(sol['x'][self.model_casadi.n_states * (self.N + 1):], 
                            self.model_casadi.n_controls, self.N)
        x = ca.reshape(sol['x'][: self.model_casadi.n_states * (self.N+1)], 
                                self.model_casadi.n_states, self.N+1)

        return x, u
        
    def get_solution(self, x0:np.ndarray, xF:np.ndarray, u0:np.ndarray) -> np.ndarray:
        """
        This function solves the optimization problem and returns the solution
        in a dictionary format based on the state and control variables
        """
        solution = self.solve(x0, xF, u0)
        x, u = self.unpack_solution(solution)
        solution_results = {
            'x': x[0,:].full().T,
            'y': x[1,:].full().T,
            'z': x[2,:].full().T,
            'phi': x[3,:].full().T,
            'theta': x[4,:].full().T,
            'psi': x[5,:].full().T,
            'u_phi': u[0,:].full().T,
            'u_theta': u[1,:].full().T,
            'u_psi': u[2,:].full().T,
            'v_cmd': u[3,:].full().T
        }
                    
        return solution_results
    
    def init_optimization_problem(self):
        self.update_bound_constraints()
        self.cost = self.compute_total_cost()
        self.init_solver(self.cost)
        print('Optimization problem initialized')
        

mpc_params = {
    'N': 20,
    'Q': ca.diag([0.1, 0.1, 0.0, 0, 0, 0]),
    'R': ca.diag([0, 0, 0, 0]),
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
    'v_cmd_max':   5.5
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
                        0, #z
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

USE_BASIC = False
USE_OBS_AVOID = False
USE_DYNAMIC_THREATS = True

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
    N_obstacles = 2
    obs_x = np.random.uniform(3, 8, N_obstacles)
    obs_y = np.random.uniform(3, 8, N_obstacles)
    random_radii = np.random.uniform(0.5, 0.9, N_obstacles)

    obs_avoid_params = {
        'weight': 1E-6,
        'safe_distance': 0.1,
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
    dynamic_threats = []
    threat_x_positions = [3] #[2, 0]
    threat_y_positions = [0] #[0, 5]
    final_position = np.array([10, 10, np.deg2rad(225)])
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
        'weight': 150,
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
        
    # ax.scatter(threat.x_traj, threat.y_traj, c=time_color, cmap='viridis', marker='o', 
    #            label='Threat trajectory')

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
    
    plt.show()
