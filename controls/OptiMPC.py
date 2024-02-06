import casadi as ca

class OptiCasadi:
    def __init__(self, mpc_params:dict,
                 casadi_model) -> None:
        
        self.opti = ca.Opti()
        self.mpc_params = mpc_params
        self.casadi_model = casadi_model
        
        self.init_mpc_params()
        self.init_decision_variables()
        
    def init_mpc_params(self)-> None:
        """initialize the mpc parameters"""
        self.N = self.mpc_params['N']
        self.Q = self.mpc_params['Q']
        self.R = self.mpc_params['R']
        self.dt = self.mpc_params['dt']
                        
    def init_decision_variables(self) -> None:
        self.X = self.opti.variable(self.casadi_model.n_states, self.N+1)
        self.U = self.opti.variable(self.casadi_model.n_controls, self.N)
        
    def set_cost_function(self) -> None:
        pass
    
    def set_constraints(self) -> None:
        pass
    
    def set_init_constraints(self) -> None:
        pass
    
    def set_terminal_constraints(self) -> None:
        pass
    
    def set_state_constraints(self) -> None:
        pass
    
    def set_control_constraints(self) -> None:
        pass    
     
    


