import casadi as ca
import time
import numpy as np
from os import system
"""
quad index
# state index
kPosX = 0
kPosY = 1
kPosZ = 2
kQuatW = 3
kQuatX = 4
kQuatY = 5
kQuatZ = 6
kVelX = 7
kVelY = 8
kVelZ = 9

# action index
kThrust = 0
kWx = 1
kWy = 2
kWz = 3
"""

class MPC():
    def __init__(self, T, dt, so_path) -> None:
        self.so_path = so_path
        
        #time constants
        self.T = T
        self.dt = dt
        self.N = int(self.T/self.dt)

        self.n_states = 3
        self.n_controls = 2

        self.Q = np.diag([1, 1, 1])
        self.R = np.diag([1, 1])

        #initial states
        self.x0 = [0, 0, 0] #x, y, psi
        self.u0 = [0, 0] #v_cmd, psi_cmd

        self.vel_min = -1.0
        self.vel_max = 1.0

        self.psi_min = np.deg2rad(-30)
        self.psi_max = np.deg2rad(30)

    def initDynamics(self):

        #states
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        psi = ca.SX.sym("psi")
        self.states = ca.vertcat(x, y, psi)

        #controls
        v_cmd = ca.SX.sym("v_cmd")
        psi_cmd = ca.SX.sym("psi_cmd")
        self.controls = ca.vertcat(v_cmd, psi_cmd)

        #dynamics
        x_dot = v_cmd * ca.cos(psi)
        y_dot = v_cmd * ca.sin(psi)
        psi_dot = psi_cmd
        self.z_dot = ca.vertcat(x_dot, y_dot, psi_dot)

        self.f = ca.Function("f", 
                             [self.states, self.controls], 
                             [self.z_dot], 
                             ["x", "u"],
                             ["x_dot"])
        
        ## Fold the last state into the first state
        F = self.sysDynamics()
        fMap = F.map(self.N, "openmp") #parallelize

        #lost function
        Delta_s = ca.SX.sym("Delta_s", self.n_states)
        Delta_p = ca.SX.sym("Delta_p", self.n_states)
        Delta_u = ca.SX.sym("Delta_u", self.n_controls)

        cost_goal = Delta_s.T @ self.Q @ Delta_s
        cost_path = Delta_p.T @ self.Q @ Delta_p
        cost_u = Delta_u.T @ self.R @ Delta_u

        f_cost_goal = ca.Function("f_cost_goal",
                                  [Delta_s],
                                    [cost_goal],
                                    ["Delta_s"],
                                    ["cost_goal"])
        
        f_cost_path = ca.Function("f_cost_path",
                                    [Delta_p],
                                    [cost_path],
                                    ["Delta_p"],
                                    ["cost_path"])
        
        f_cost_u = ca.Function("f_cost_u",
                                [Delta_u],
                                [cost_u],
                                ["Delta_u"],
                                ["cost_u"])
        
        ##nonlinear optimization problem
        self.nlp_w = []       # nlp variables
        self.nlp_w0 = []      # initial guess of nlp variables
        self.lbw = []         # lower bound of the variables, lbw <= nlp_x
        self.ubw = []         # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0      # objective 
        self.nlp_g = []       # constraint functions
        self.lbg = []         # lower bound of constraint functions, lbg < g
        self.ubg = []         # upper bound of constraint functions, g < ubg

        u_min = [self.vel_min, self.psi_min]
        u_max = [self.vel_max, self.psi_max]

        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self.n_states)]
        x_max = [+x_bound for _ in range(self.n_states)]
        #
        g_min = [0 for _ in range(self.n_states)]
        g_max = [0 for _ in range(self.n_states)]

        # P = ca.SX.sym("P", self.n_states+(self.n_states+3)*self.N+self.n_states)
        X = ca.SX.sym("X", self.n_states, self.N+1)
        U = ca.SX.sym("U", self.n_controls, self.N)
        
        X_next = fMap(X[:, :self.N], U) 

        # Lift initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += [self.x0]
        self.lbw += self.x0
        self.ubw += self.x0

        #starting point
        # self.nlp_g += [X[:, 0] - P[:self.n_states]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self.N):
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += [self.u0]
            self.lbw += u_min
            self.ubw += u_max

            #get time constants
            idx_k = self.n_states + (self.n_states+3)*k
            idx_k_end = self.n_states + (self.n_states+3)*(k+1)
            # time_k = P[idx_k:idx_k_end]

            weight = 1

            # cost for tracking the goal position
            cost_goal_k, cost_gap_k = 0, 0
            # if k >= self.N-1: # The goal postion.
            #     delta_s_k = (X[:, k+1] - P[self.n_states+(self.n_states+3)*self.N:])
            #     cost_goal_k = f_cost_goal(delta_s_k)
            # else:
            #     # cost for tracking the moving gap
            #     delta_p_k = (X[:, k+1] - P[self.n_states+(self.n_states+3)*k : \
            #         self.n_states+(self.n_states+3)*(k+1)-3]) 
            #     cost_gap_k = f_cost_path(delta_p_k) * weight 
            
            #cost from goal position
            # delta_s_k = (X[:, k] - self.goal)
            # cost_goal_k = f_cost_goal(delta_s_k)
            
            delta_u_k = U[:, k] - U[:, k-1]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k + cost_gap_k

            #new NLP variable for state at end of interval
            self.nlp_w += [X[:, k+1]]
            self.nlp_w0 += [self.x0]
            self.lbw += x_min
            self.ubw += x_max

            #add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max

        #create nlp solver
        nlp_dict = {'f': self.mpc_obj,
                    'x': ca.vertcat(*self.nlp_w),
                    # 'p': P,
                    'g': ca.vertcat(*self.nlp_g)}


        ##ipopt solver
        ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "ipopt.linear_solver": "ma27",
            "print_time": False
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)
        # jit compile for speed up
        print("Generating shared library........")
        cname = self.solver.generate_dependencies("nmpc_v0.c")  
        system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path) # -O3

        #reload compiled mpc
        self.solver = ca.nlpsol("solver", "ipopt", self.so_path , ipopt_options)
        print("got solver")

    def solve(self,ref_states):
        
        #ref_states = ca.vertcat(*ref_states)
        
        self.sol = self.solver(
            x0=self.nlp_w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            p=ref_states, 
            lbg=self.lbg, 
            ubg=self.ubg)
        

        sol_x0 = self.sol["x"].full()
        opt_u = sol_x0[self.self.n_states:self.self.n_states+self.n_controls]

        #warm start
        self.nlp_w0 = list(sol_x0[self.n_states+self.n_controls:2*(self.n_states+self.n_controls)]) + \
            list(sol_x0[self.n_states+self.n_controls:])

        x0_array = np.reshape(sol_x0[:-self.n_states], newshape=(-1, self.n_states+self.n_controls))

        return opt_u, x0_array
    

    def sysDynamics(self):
        M = 4 #refinement
        DT = self.dt/M
        X0 = ca.SX.sym("X0", self.n_states)
        U = ca.SX.sym("U", self.n_controls)
        X = X0
        for _ in range(M):
            k1 = DT * self.f(X, U)
            k2 = DT * self.f(X + k1/2, U)
            k3 = DT * self.f(X + k2/2, U)
            k4 = DT * self.f(X + k3, U)
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6

        #fold the last state into the first state
        F = ca.Function("F", [X0, U], 
                        [X], 
                        ["x0", "u"], 
                        ["xf"])
        return F

if __name__ == "__main__":
    
    t_horizon = 2.0 #prediction horizon
    dt = 0.1 #sampling time
    so_path = "nmpc_v0.so"

    mpc = MPC(t_horizon, dt, so_path)
    mpc.initDynamics()

    #initial state
    x0 = [0, 0, 0]
    xf = [10, 0, 0]
    xref = x0 + xf
    mpc.solve(xref)