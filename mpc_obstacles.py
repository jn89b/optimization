### Formulate the MPC problem with obstacles
import numpy as np
import casadi as ca

from matplotlib import pyplot as plt
from controls.OptiMPC import OptiCasadi
from models.Plane import Plane


class PlaneMPC(OptiCasadi):
    """
    Include obstacle avoidance in the formulation
    """
    def __init__(self, mpc_params: dict, 
                 control_constraints:dict,
                 state_constraints:dict,
                 control_indices:dict,
                 state_indices:dict,
                 casadi_model ) -> None:
        super().__init__(mpc_params, casadi_model)
        
        self.x0 = self.opti.parameter(self.casadi_model.n_states)
        self.u0 = self.opti.parameter(self.casadi_model.n_controls)
        self.xF = self.opti.parameter(self.casadi_model.n_states)
        
        self.control_constraints = control_constraints
        self.state_constraints = state_constraints
        
        self.control_indices = control_indices
        self.state_indices = state_indices
        
        self.init_mpc_params()
        self.init_decision_variables()
        self.set_init_constraints()
        
    def set_init_constraints(self) -> None:
        pass
        #self.opti.subject_to(self.X[:,0] == self.x0)
    
    # def set_init_final_states(self, x0:np.array, xf:np.array) -> None:
    def set_init_final_states(self, init_state:np.ndarray, 
                              final_state:np.ndarray) -> None:
        init_state = init_state.tolist()
        final_state = final_state.tolist()
        self.opti.set_value(self.x0, init_state)
        self.opti.set_value(self.xF, final_state)
    
    def set_init_controls(self, u0:np.array) -> None:
        u0 = u0.tolist()
        self.opti.set_value(self.u0, u0)
    
    def set_state_constraints(self) -> None:
        state_constraints = self.state_constraints
        # state_indicies = []
        # for key in state_constraints.keys():
        #     state_indicies.append(state_indices[key])
        
        state_indices = self.state_indices
        print('State indices', state_indices)        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['x_dot_min'],
            self.X[state_indices['x_dot'],:],
            state_constraints['x_dot_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['y_dot_min'],
            self.X[state_indices['y_dot'],:],
            state_constraints['y_dot_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['z_dot_min'],
            self.X[state_indices['z_dot'],:],
            state_constraints['z_dot_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['phi_min'],
            self.X[state_indices['phi_dot'],:],
            state_constraints['phi_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['theta_min'],
            self.X[state_indices['theta_dot'],:],
            state_constraints['theta_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['psi_min'],
            self.X[state_indices['psi_dot'],:],
            state_constraints['psi_max']))
        
        # self.opti.subject_to(self.opti.bounded(
        #     state_constraints['airspeed_min'],
        #     self.X[state_indices['airspeed'],:],
        #     state_constraints['airspeed_max']))
    
    def set_control_constraints(self) -> None:
        """
        """
        control_constraints = self.control_constraints
        control_indices = self.control_indices
        print('Control indices', control_indices)

        u_phi = self.U[control_indices['u_phi'],:]
        u_theta = self.U[control_indices['u_theta'],:]
        u_psi = self.U[control_indices['u_psi'],:]
        v_cmd = self.U[control_indices['v_cmd'],:]
        
        self.opti.subject_to(self.opti.bounded(
            control_constraints['u_phi_min'],
            u_phi,
            control_constraints['u_phi_max']))
        
        self.opti.subject_to(self.opti.bounded(
            control_constraints['u_theta_min'],
            u_theta,
            control_constraints['u_theta_max']))
        
        self.opti.subject_to(self.opti.bounded(
            control_constraints['u_psi_min'],
            u_psi,
            control_constraints['u_psi_max']))
        
        self.opti.subject_to(self.opti.bounded(
            control_constraints['v_cmd_min'],
            v_cmd,
            control_constraints['v_cmd_max']))
    
    def set_dynamic_constraint(self) -> None:
        self.opti.subject_to(self.X[:,0] == self.x0)
        
        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]
                
            ##Runge Kutta
            k1 = self.casadi_model.function(states, controls)
            k2 = self.casadi_model.function(states + self.dt/2*k1, controls)
            k3 = self.casadi_model.function(states + self.dt/2*k2, controls)
            k4 = self.casadi_model.function(states + self.dt * k3, controls)
            state_next_RK4 = states + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(state_next == state_next_RK4)

    def set_cost_function(self, x_final:np.ndarray) -> None:
        Q = self.mpc_params['Q']
        R = self.mpc_params['R']
        # cost = 0
        x_position = self.X[0, :]
        y_position = self.X[1, :]
        
        #check if x_final is a numpy array
        if isinstance(x_final, np.ndarray):
           #turn into list
              x_final = x_final.tolist()
               
        error_x = x_final[0] - x_position
        error_y = x_final[1] - y_position
        
        sum_ex = ca.sumsqr(error_x) * 1
        sum_ey = ca.sumsqr(error_y) * 1
        
        cost = sum_ex + sum_ey
        print('Cost function set', cost)
        self.opti.minimize(cost)            

    def init_optimal_control_prob(self, start_states:np.array,
                                end_states:np.array) -> None:
        self.set_init_final_states(start_states, end_states)
        self.set_control_constraints()
        self.set_state_constraints()
        self.set_dynamic_constraint()
        self.set_cost_function(end_states)
        self.set_solution_options()
        print('Optimal control problem initialized')

    def solve(self) -> ca.OptiSol:
        sol = self.opti.solve()
        return sol


def format_mpc_trajectory(solution:ca.OptiSol, 
                          state_idx_dict:dict, 
                          control_idx_dict:dict) -> tuple:
    states   = solution.value(mpc_controller.X)
    controls = solution.value(mpc_controller.U)
    
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
    'N': 30,
    'Q': np.diag([1, 1, 1, 1, 1, 1, 1]),
    'R': np.diag([1, 1, 1, 1]),
    'dt': 0.05
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(15),
    'u_theta_max': np.deg2rad(15),
    'u_psi_min':  -np.deg2rad(30),
    'u_psi_max':   np.deg2rad(30),
    'v_cmd_min':   5,
    'v_cmd_max':   10
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
    'airspeed_max': 5
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

init_states = np.array([2, #x 
                        2, #y
                        2, #z
                        0, #phi
                        0, #theta
                        0, #psi# 3  #airspeed
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
                          8])

plane = Plane()
plane.set_state_space()

mpc_controller = PlaneMPC(mpc_params, 
                          control_constraints, 
                          state_constraints,
                          control_indices,
                          state_indices,
                          plane)

mpc_controller.set_init_controls(init_controls)
mpc_controller.init_optimal_control_prob(init_states, final_states)

solution = mpc_controller.solve()
state_traj   = solution.value(mpc_controller.X)
control_traj   = solution.value(mpc_controller.U)

state_dict, control_dict = format_mpc_trajectory(solution,
                                                    state_indices,
                                                    control_indices)

fig,ax = plt.subplots(1, figsize=(10,10))
ax.plot(state_dict['x_dot'], state_dict['y_dot'], 'r')

# plt.show()