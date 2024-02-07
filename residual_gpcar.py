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
import pickle as pkl


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
dt = 0.05

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

num_samples = 100
LOAD_DATA = True
SAVE = True
car = ToyCar()
car.set_state_space()

##### Generate Trainig Data and build the GP model #####
if LOAD_DATA:
    Z_train = pkl.load(open("Z_train.pkl", "rb"))
    Y_train = pkl.load(open("Y_train.pkl", "rb"))
    Z_train = torch.tensor(Z_train).float()
    Y_train = torch.tensor(Y_train).float()
else:
    Z_train, Y_noise ,Y_nominal = car.generate_training_data(num_samples, dt, constraint_params, 
                                                            noise_val=0.5, noise_deg=0.5)
    Z_train = torch.tensor(Z_train).float()
    Y_train = Y_noise - Y_nominal
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

training_iter = 100
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
final_states = np.array([0, 5, 0])

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
n_steps = 300 

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
    'gp_upper': [],
    'true_state': [],
    'error_prediction': []
}

z_test_array = []

noise_val = 1e-4
for i in range(n_steps):
    unexplained_psi_bias = np.random.uniform(np.deg2rad(6), np.deg2rad(8))
    solution = mpc_controller.solve()
    x_traj = solution.value(mpc_controller.X)
    u_traj = solution.value(mpc_controller.U)
    
    #reinit here 
    t_init, init_vector = shift_time_step(dt, t_init, init_vector, u_traj, car.function)
      
    #add random noise to the state
    #make a copy of the true state
    true_state     = np.copy(init_vector)
    true_state[0] += np.random.uniform(-noise_val, noise_val)
    true_state[1] += np.random.uniform(-noise_val, noise_val)
    #add noise to the psi state
    noise_rad = np.random.uniform(-np.deg2rad(0.1), np.deg2rad(0.1))
    true_state[2] += noise_rad + unexplained_psi_bias
    # true_state += np.random.normal(-noise_val, noise_val, init_vector.shape)
    
    if i % print_every == 0:
        print("init vector is ", init_vector)

    post_results['solution_list'].append((x_traj, u_traj))
    mpc_controller.opti.set_value(mpc_controller.x0, init_vector)

    final_states[0] += 1
    final_states[1] += 1
    
    mpc_controller.opti.set_value(mpc_controller.xF, final_states)
    post_results['time_history'].append(t_init)
    
    # Predict the next state with GP model
    #combine the combine the state and control inputs
    u_init = np.array(u_traj[:,0])
    #shape u_init as a row vector
    u_init = u_init.reshape(1, -1)
    Z_test = np.hstack((np.transpose(init_vector), u_init))
    Z_test = torch.tensor(Z_test).float().to(device)
    z_test_array.append(Z_test)
    
    dist_from_goal = np.linalg.norm(init_vector[:-1] - final_states[:-1])
    post_results['state_history'].append(init_vector)
    post_results['control_history'].append(u_init)
    post_results['true_state'].append(true_state)
    
    error_prediction = true_state - init_vector
    post_results['error_prediction'].append(error_prediction)
    
    # init_vector = true_state
    
    if dist_from_goal < tolerance:
        print("Goal reached")
        break 

error = 0
## output would be the residuals of the system 
## we want to see how well the GP model can predict the residuals
gp_pred_x = []
gp_pred_y = []
gp_pred_psi = []
for i, z in enumerate(z_test_array):
    
    predictions = likelihood(model(z))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    mean = mean.cpu().detach().numpy()
    lower = lower.cpu().detach().numpy()
    upper = upper.cpu().detach().numpy()
    
    gp_x = mean[0][0] + post_results['state_history'][i][0]
    gp_y = mean[0][1] + post_results['state_history'][i][1]
    gp_psi = mean[0][2] + post_results['state_history'][i][2]
    gp_pred_x.append(gp_x)
    gp_pred_y.append(gp_y)
    gp_pred_psi.append(gp_psi)

    post_results['gp_predictions'].append(mean)
    post_results['gp_lower'].append(lower)
    post_results['gp_upper'].append(upper)
    
    diff_state = np.array(post_results['true_state'][i]) - np.array(post_results['state_history'][i])
    
    #compute the error
    error += np.linalg.norm(mean - diff_state)
        
mse = error / len(z_test_array)
print("MSE is ", mse)
print("final state is ", init_vector)
print("final time is ", t_init)

#%% Post Processing 
x_history = []
y_history = []
psi_history = []

error_x_history = []
error_y_history = []
error_psi_history = []

true_x_history = []
true_y_history = []
true_psi_history = []

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
    
for state in post_results['true_state']:
    true_x_history.append(state[0])
    true_y_history.append(state[1])
    true_psi_history.append(state[2])

for i in range(len(post_results['state_history'])):
    error_x_history.append(post_results['true_state'][i][0] - post_results['state_history'][i][0])
    error_y_history.append(post_results['true_state'][i][1] - post_results['state_history'][i][1])
    error_psi_history.append(post_results['true_state'][i][2] - post_results['state_history'][i][2])
     

#%% Plot stuff
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.plot(x_history, y_history, label='Nominal', linestyle='-', marker='o', color='b')
ax.plot(gp_pred_x, gp_pred_y, label='GP', linestyle='-', marker='o', color='r')
ax.plot(true_x_history, true_y_history, label='True', linestyle='-', marker='o', color='g')
#ax.fill_between(np.arange(len(x_history)), gp_lower[:,0], gp_upper[:,0], alpha=0.5)
ax.set_title('X vs Y')
ax.legend()

fig,ax = plt.subplots(3, 1, figsize=(14, 10))
ax[0].plot(post_results['time_history'], error_x_history, linestyle='--', marker='s', label='X Error', color='b')
ax[0].plot(post_results['time_history'], gp_predictions[:,0], linestyle='--', marker='s', label='GP', color='r')
ax[0].fill_between(post_results['time_history'], gp_lower[:,0], gp_upper[:,0], alpha=0.5)

ax[1].plot(post_results['time_history'], error_y_history, linestyle='--', marker='s', label='Y Error' , color='b')
ax[1].plot(post_results['time_history'], gp_predictions[:,1], linestyle='--', marker='s', label='GP', color='r')
ax[1].fill_between(post_results['time_history'], gp_lower[:,1], gp_upper[:,1], alpha=0.5)

ax[2].plot(post_results['time_history'], error_psi_history, linestyle='--', marker='s', label='Psi Error', color='b')
ax[2].plot(post_results['time_history'], gp_predictions[:,2], linestyle='--', marker='s', label='GP', color='r')
ax[2].fill_between(post_results['time_history'], gp_lower[:,2], gp_upper[:,2], alpha=0.5)

ax[0].set_title('X Error Prediction')
ax[1].set_title('Y Error Prediction')
ax[2].set_title('Psi Error Prediction')

#save the test dataset 
#convert to z_test to tensor
if SAVE:
    z_test_array = torch.stack(z_test_array)
    z_test_array = z_test_array.reshape(-1, car.n_states + car.n_controls)
    #convert to cpu
    z_test_array = z_test_array.cpu().numpy()
    pkl.dump(z_test_array, open("Z_train.pkl", "wb"))

    #convert x_history, y_history, psi_history to tensor
    x_history = torch.tensor(error_x_history)
    y_history = torch.tensor(error_y_history)
    psi_history = torch.tensor(error_psi_history)

    y_true = torch.stack([x_history, y_history, psi_history], dim=1)
    y_true = y_true.reshape(-1, car.n_states)
    y_true = y_true.cpu().numpy()
    pkl.dump(y_true, open("Y_train.pkl", "wb"))

#plot the titl

for a in ax:
    a.legend()
    a.grid(True)
    
plt.show()