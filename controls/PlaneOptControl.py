from controls.OptimalControl import OptimalControlProblem
import casadi as ca
import numpy  as np


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
        
        self.is_initialized = False  
        
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
        total_threat_cost = 0
        threat_cost = 0
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
            
            time_index = 1
            
            for i in range(self.N):
                
                for j in range(time_index):
                    idx = j + i
                    if idx > self.N:
                        break
                    #calculate the distance between the ego vehicle and the threat
                    distance = ca.sqrt((x_pos[i] - threat_x_traj[idx])**2 + \
                        (y_pos[i] - threat_y_traj[idx])**2)
                    #avoid division by zero
                    #check if the distance vector has any negative values
                    # ca.if_else(distance>0, 1, distance)
                    
                    #get unit vector of ego vehicle
                    u_x = ca.cos(psi[i])
                    u_y = ca.sin(psi[i])
                    #get unit vector of threat
                    u_x_t = ca.cos(threat_psi_traj[idx])
                    u_y_t = ca.sin(threat_psi_traj[idx])
                    
                    #calculate the dot product of the unit vectors
                    dot_product = (u_x * u_x_t) + (u_y * u_y_t)
                    
                    #if the value of the difference becomes positive that means we are in danger
                    # ca.if_else(distance<threat_radii, 1, distance)
                    distance_cost = 1/distance

                    #add the cost to the total cost
                    # threat_cost += self.dynamic_threat_params['weight'] * ca.sumsqr(distance_cost) + \
                    #     self.dynamic_threat_params['weight'] * ca.sumsqr(dot_product)
                    threat_cost +=  dot_product
                    
            total_threat_cost = self.dynamic_threat_params['weight'] * ca.sumsqr(threat_cost)

            # distance = ca.sqrt((x_pos.T - threat_x_traj)**2 + \
            #     (y_pos.T - threat_y_traj)**2)


            # # distance = ca.sqrt((x_pos.T - threat_x_traj)**2 + \
            # #     (y_pos.T - threat_y_traj))**2
            # threat_radii = 5.0
            # safe_distance =  1.0
            # #avoid division by zero
            # #check if the distance vector has any negative values
            # ca.if_else(distance>threat_radii, 1, distance)
                    
            # #if the value of the difference bec
            # # omes positive that means we are in danger
            # #distance_cost = -distance + threat_radii + safe_distance
            # distance_cost = 1/distance
            

            # #add the cost to the total cost
            # threat_cost = self.dynamic_threat_params['weight'] * ca.sumsqr(distance_cost) 
                 
            # cost += threat_cost
            
        return threat_cost
    
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
            'x': x[0,:].full().T[:,0],
            'y': x[1,:].full().T[:,0],
            'z': x[2,:].full().T[:,0],
            'phi': x[3,:].full().T[:,0],
            'theta': x[4,:].full().T[:,0],
            'psi': x[5,:].full().T[:,0],
            'u_phi': u[0,:].full().T[:,0],
            'u_theta': u[1,:].full().T[:,0],
            'u_psi': u[2,:].full().T[:,0],
            'v_cmd': u[3,:].full().T[:,0]
        }
                    
        return solution_results
    
    def init_optimization_problem(self):
        if self.is_initialized:
            print('Optimization problem already initialized')
            self.g = []
            self.cost = 0
            
        self.update_bound_constraints()
        self.cost = self.compute_total_cost()
        self.init_solver(self.cost)
        self.is_initialized = True
        print('Optimization problem initialized')
        
        
    def update_dynamic_threats(self, threat_params:dict) -> None:
        if self.is_initialized:
            print('Optimization problem already initialized')
            self.g = []
            self.cost = 0
            
            
        self.dynamic_threat_params = threat_params
        self.use_dynamic_threats = True
        self.cost = self.compute_total_cost()
        self.init_solver(self.cost)
        self.is_initialized = True
        print('Dynamic threats updated')
    
    
    