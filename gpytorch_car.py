"""
Steps:
- Build toy car
- Build PID system 
- Build GP model

Simulate the car with the PID system
- Have measurements of the car's state with noise
- Use the GP model to predict the next state of the car
- Use the MPC to generate the next control input
- Repeat

- Compare to the car with no GP model, that is using nominal values
"""

import numpy as np
import torch
import gpytorch
import casadi as ca # this will be done eventually

from matplotlib import pyplot as plt
from models.toycar import ToyCar
from gaussian_processes.gpytorch_interface import MultitaskGPModel
from controls.OptiMPC import OptiCasadi

# inherit from OptiMPC
class GPMPController(OptiCasadi):
    def __init__(self, mpc_params:dict, control_constraints:dict, 
                 casadi_model):
        super().__init__(
            mpc_params, casadi_model)
        
        self.x0 = self.opti.parameter(self.casadi_model.n_states)
        self.xF = self.opti.parameter(self.casadi_model.n_states)
        self.control_constraints = control_constraints
    
    def set_init_final_states(self, init_state:np.ndarray, final_state:np.ndarray) -> None:
        self.opti.set_value(self.x0, init_state)
        self.opti.set_value(self.xF, final_state)
    
    def set_control_constraints(self) -> None:
        control_constraints = self.control_constraints
        
        vel_min = control_constraints['lower_input_bounds'][0]
        psi_rate_min = control_constraints['lower_input_bounds'][1]
        
        vel_max = control_constraints['upper_input_bounds'][0]
        psi_rate_max = control_constraints['upper_input_bounds'][1]
        
        self.opti.subject_to(self.opti.bounded(
            vel_min, 
            self.U[0,:], 
            vel_max
        ))
        
        #set control rate limits
        self.opti.subject_to(self.opti.bounded(
            psi_rate_min,
            self.U[1,:],
            psi_rate_max
        ))

        
        print("control constraints set")
        
    def set_dynamic_constraint(self) -> None:
        self.opti.subject_to(self.X[:,0] == self.x0)
        for k in range(self.N):
            current_state = self.X[:, k]
            current_control = self.U[:,k]
            k1 = self.casadi_model.function(current_state, current_control)
            k2 = self.casadi_model.function(current_state + self.dt/2 * k1, current_control)
            k3 = self.casadi_model.function(current_state + self.dt/2 * k2, current_control)
            k4 = self.casadi_model.function(current_state + self.dt * k3, current_control)
            state_next_rk4 = current_state + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            self.opti.subject_to(self.X[:, k+1] == state_next_rk4)
        
    def set_cost_function(self, x_final:np.ndarray) -> None:
        Q = self.mpc_params['Q']
        R = self.mpc_params['R']
        

        cost = 0
        x_position = self.X[0,:]
        y_position = self.X[1,:]
        
        #check if x_final is a numpy array
        if isinstance(x_final, np.ndarray):
           #turn into list
              x_final = x_final.tolist()
               
        error_x = x_final[0] - x_position
        error_y = x_final[1] - y_position
        
        sum_ex = ca.sumsqr(error_x) * 1
        sum_ey = ca.sumsqr(error_y) * 1
        # mag_error = ca.sqrt(sum_ex + sum_ey)
        cost = sum_ex + sum_ey
        
        self.opti.minimize(cost)
    
    def set_solution_options(self, print_time:int=0) -> None:
        opts = {
            'ipopt': {
                'max_iter': 5000,
                'print_level': 1,
                'acceptable_tol': 1e-2,
                'acceptable_obj_change_tol': 1e-2,
            },
            'print_time': print_time
        }
        
        self.opti.solver('ipopt', opts)#, {'ipopt': {'print_level': 0}})
        

    def solve(self) -> ca.OptiSol:
        sol = self.opti.solve()
        return sol


def shift_time_step(step_horizon, t_init, state_init, u,f):
    f_value = f(state_init, u[:,0]) #calls out the runge kutta
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    #shift time and controls
    next_t = t_init + step_horizon
    return next_t, next_state


##### Define the Car Model #####

# System parameters for car
dt = 0.1

# Lower and upper bounds for states and inputs
xlb = np.array([-10, -10, -np.pi])
xub = np.array([10, 10, np.pi])
ulb = np.array([0.5, -np.deg2rad(45)])
uub = np.array([1,  np.deg2rad(45)])

constraint_params = {
    "lower_state_bounds": xlb,
    "upper_state_bounds": xub,
    "lower_input_bounds": ulb,
    "upper_input_bounds": uub,
}

num_samples = 500

car = ToyCar()
car.set_state_space()

##### Generate Trainig Data and build the GP model #####

Z_train, Y_train = car.generate_training_data(num_samples, dt, constraint_params)
Z_train = torch.tensor(Z_train).float()
Y_train = torch.tensor(Y_train).float()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=car.n_states)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)
Z_train = Z_train.to(device)
Y_train = Y_train.to(device)
model = MultitaskGPModel(Z_train, Y_train, likelihood, car.n_states).to(device)
likelihood = likelihood.to(device)

##### Train the GP model #####
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 25
mse_list = []
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(Z_train)
    loss = -mll(output, Y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    mse_list.append(loss.item())
    
# begin simulation of real life car
# Set into eval mode
model.eval()
likelihood.eval()

##### Simulate the car MPC #####
Q = np.eye(car.n_states) 
R = np.eye(car.n_controls)

mpc_params = {
    'N': 50,
    'Q': Q,
    'R': R,
    'dt': dt
}

init_states = np.array([0, 0, 0])
final_states = np.array([4, 4, 0])

mpc_controller = GPMPController(mpc_params,constraint_params, car)
mpc_controller.set_init_final_states(init_states, final_states)
mpc_controller.set_control_constraints()
mpc_controller.set_dynamic_constraint()
mpc_controller.set_cost_function(final_states)
mpc_controller.set_solution_options()

solution = mpc_controller.solve()
x_traj = solution.value(mpc_controller.X)
u_traj = solution.value(mpc_controller.U)\


#### Begin Simulation of the car with the GP model ####
n_steps = 50

t_init = 0

dist_from_goal = np.linalg.norm(init_states[:-1] - final_states[:-1])
print("distance from goal is ", dist_from_goal)

tolerance = 0.1
init_vector = init_states
print_every = 5

post_results = {
    'solution_list': [],
    'time_history': [],
    'state_history': [],
    'control_history': [],
    'gp_predictions': [],
    'gp_lower': [],
    'gp_upper': []
}

for i in range(n_steps):
        
    solution = mpc_controller.solve()
    x_traj = solution.value(mpc_controller.X)
    u_traj = solution.value(mpc_controller.U)
    
    #reinit here 
    t_init, init_vector = shift_time_step(dt, t_init, init_vector, u_traj, car.function)
    
    if i % print_every == 0:
        print("init vector is ", init_vector)

    post_results['solution_list'].append((x_traj, u_traj))
    mpc_controller.opti.set_value(mpc_controller.x0, init_vector)

    #get the next control input
    mpc_controller.opti.set_value(mpc_controller.x0, init_vector)
    post_results['time_history'].append(t_init)
    
    # Predict the next state with GP model
    #combine the combine the state and control inputs
    u_init = np.array(u_traj[:,0])
    #shape u_init as a row vector
    u_init = u_init.reshape(1, -1)
    Z_test = np.hstack((np.transpose(init_vector), u_init))
    Z_test = torch.tensor(Z_test).float().to(device)
    predictions = likelihood(model(Z_test))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    mean = mean.cpu().detach().numpy()
    lower = lower.cpu().detach().numpy()
    upper = upper.cpu().detach().numpy()
    print("mean is ", mean)
    print("lower is ", lower)
    print("upper is ", upper)
    
    dist_from_goal = np.linalg.norm(init_vector - final_states[:-1])
    
    post_results['gp_predictions'].append(mean)
    post_results['gp_lower'].append(lower)
    post_results['gp_upper'].append(upper)
    post_results['state_history'].append(init_vector)
    post_results['control_history'].append(u_init)
    
    if dist_from_goal < tolerance:
        print("Goal reached")
        break 

print("final state is ", init_vector)
print("final time is ", t_init)

#%% Post Processing 
x_history = []
y_history = []
psi_history = []

gp_predictions = post_results['gp_predictions']
gp_predictions = np.array(gp_predictions)
gp_predictions = gp_predictions.reshape(-1, car.n_states)

gp_lower = post_results['gp_lower']
gp_lower = np.array(gp_lower)
gp_lower = gp_lower.reshape(-1, car.n_states)

gp_upper = post_results['gp_upper']
gp_upper = np.array(gp_upper)
gp_upper = gp_upper.reshape(-1, car.n_states)

for state in post_results['state_history']:
    x_history.append(state[0])
    y_history.append(state[1])
    psi_history.append(state[2])
    
#%% Plot stuff
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.scatter(x_history, y_history, label='Nominal')
ax.plot(gp_predictions[:,0], gp_predictions[:,1], 
        label='GP', linestyle='--', color='r')
#ax.fill_between(np.arange(len(x_history)), gp_lower[:,0], gp_upper[:,0], alpha=0.5)
ax.set_title('X vs Y')
ax.legend()


fig,ax = plt.subplots(3, 1, figsize=(14, 10))
ax[0].scatter(post_results['time_history'], x_history, label='Nominal')
ax[0].plot(post_results['time_history'], gp_predictions[:,0], label='GP', linestyle='--', color='r')
ax[0].fill_between(post_results['time_history'], gp_lower[:,0], gp_upper[:,0], alpha=0.5)

ax[1].scatter(post_results['time_history'], y_history, label='Nominal')
ax[1].plot(post_results['time_history'], gp_predictions[:,1], label='GP', linestyle='--', color='r')
ax[1].fill_between(post_results['time_history'], gp_lower[:,1], gp_upper[:,1], alpha=0.5)

ax[2].scatter(post_results['time_history'], psi_history, label='Nominal')
ax[2].plot(post_results['time_history'], gp_predictions[:,2], label='GP', linestyle='--', color='r')
ax[2].fill_between(post_results['time_history'], gp_lower[:,2], gp_upper[:,2], alpha=0.5)

ax[0].set_title('X State')
ax[1].set_title('Y State')
ax[2].set_title('Psi State')

for a in ax:
    a.legend()
    a.grid(True)
    
plt.show()