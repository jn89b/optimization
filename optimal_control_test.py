
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
                 mpc_params:dict, casadi_model):
        super().__init__(mpc_params, casadi_model)
        self.control_constraints = control_constraints
        self.state_constraints = state_constraints
        # self.update_bound_constraints()
        self.cost = 0
        # self.x0 = self.opti.parameter(self.casadi_model.n_states)
        # self.xF = self.opti.parameter(self.casadi_model.n_states)
        # self.control_constraints = control_constraints
        
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

    def compute_total_cost(self) -> ca.SX:
        self.cost = self.compute_dynamics_cost()
    
    def init_optimization_problem(self):
        self.update_bound_constraints()
        self.cost = self.compute_dynamics_cost()
        self.init_solver(self.cost)
        print('Optimization problem initialized')
        
    def get_solution(self, x0:np.ndarray, xF:np.ndarray, u0:np.ndarray) -> np.ndarray:
        return self.solve(x0, xF, u0)
    
mpc_params = {
    'N': 20,
    'Q': ca.diag([1, 1, 1, 1, 1, 1]),
    'R': ca.diag([1, 1, 1, 1]),
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

plane = Plane()
plane.set_state_space()

plane_mpc = PlaneOptControl(
    control_constraints, 
    state_constraints, 
    mpc_params, 
    plane
)

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


# plane_mpc.compute_dynamics_cost()
# plane_mpc.init_solver()
plane_mpc.init_optimization_problem()
solve = plane_mpc.get_solution(init_states, final_states, init_controls)