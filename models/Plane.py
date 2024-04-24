import numpy as np

import casadi as ca
from matplotlib import pyplot as plt
from os import system

class Plane():
    def __init__(self, 
                 include_time:bool=False,
                 dt_val:float=0.05,
                 compile_to_c:bool=False,
                 use_compiled_fn:bool=False) -> None:
        self.include_time = include_time
        self.dt_val = dt_val
        self.compile_to_c = compile_to_c
        self.use_compiled_fn = use_compiled_fn
        self.define_states()
        self.define_controls() 
        
    def define_states(self):
        """define the states of your system"""
        #positions ofrom world
        self.x_f = ca.MX.sym('x_f')
        self.y_f = ca.MX.sym('y_f')
        self.z_f = ca.MX.sym('z_f')

        #attitude
        self.phi_f = ca.MX.sym('phi_f')
        self.theta_f = ca.MX.sym('theta_f')
        self.psi_f = ca.MX.sym('psi_f')
        self.v = ca.MX.sym('t')

        if self.include_time:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f, 
                self.v)
        else:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f,
                self.v 
            )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """controls for your system"""
        self.u_phi = ca.MX.sym('u_phi')
        self.u_theta = ca.MX.sym('u_theta')
        self.u_psi = ca.MX.sym('u_psi')
        self.v_cmd = ca.MX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0] 

    def set_state_space(self):
        """define the state space of your system"""
        self.g = 9.81 #m/s^2
        #body to inertia frame 
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f) 
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)
        
        self.phi_fdot   = self.u_phi 
        self.theta_fdot = self.u_theta
        
        #check if the denominator is zero
        # self.v_cmd = ca.if_else(self.v_cmd == 0, 1e-6, self.v_cmd)
        self.v_dot = ca.sqrt(self.x_fdot**2 + self.y_fdot**2 + self.z_fdot**2)
        self.psi_fdot   = self.u_psi + (self.g * (ca.tan(self.phi_f) / self.v_cmd))

        # self.t_dot = self.t 
        
        if self.include_time:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
                self.psi_fdot,
                self.v_dot
            )
        else:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
                self.psi_fdot,
                self.v_dot
            )

        #ODE function
        name = 'dynamics'
        self.function = ca.Function(name, 
            [self.states, self.controls], 
            [self.z_dot])

        folder_dir = 'c_code'

        #oname = folder_dir+'/'+name + '.so'
        oname = name + '.so'
        
        if self.compile_to_c:
            function = self.function.generate()
            print("function: ", function)
            system('gcc -pipe -fPIC -shared -O3 ' +  function + ' -o ' + oname)
            # self.f = ca.external(name, './'+oname)
            print("Compiled to C")
        
        if self.use_compiled_fn:
            print("Using compiled function")
            self.c_code = ca.external(name, oname)
            self.function = self.c_code
            
        
    def rk45(self, x, u, dt, use_numeric:bool=True):
        """
        Runge-Kutta 4th order integration
        x is the current state
        u is the current control input
        dt is the time step
        use_numeric is a boolean to return the result as a numpy array
        """
        k1 = self.function(x, u)
        k2 = self.function(x + dt/2 * k1, u)
        k3 = self.function(x + dt/2 * k2, u)
        k4 = self.function(x + dt * k3, u)
        
        next_step = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        #return as numpy row vector
        if use_numeric:
            next_step = np.array(next_step).flatten()
            return next_step
        else:
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    


    
    
