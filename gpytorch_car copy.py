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

num_samples = 300

car = ToyCar()
car.set_state_space()

##### Generate Trainig Data and build the GP model #####

Z_train, Y_noise ,Y_nominal = car.generate_training_data(num_samples, dt, constraint_params, 
                                                         noise_val=0.5, noise_deg=0.5)
Z_train = torch.tensor(Z_train).float()
Y_train = Y_noise - Y_nominal
Y_train = torch.tensor(Y_train).float()

Z_test, Y_test_noise, Y_test_nominal = car.generate_training_data(100, dt, constraint_params,
                                                                    noise_val=0.5, noise_deg=0.5)

Z_test = torch.tensor(Z_test).float()
Y_test = Y_test_noise - Y_test_nominal
Y_test = torch.tensor(Y_test).float()


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=car.n_states)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)
Z_train = Z_train.to(device)
Y_train = Y_train.to(device)
Z_test = Z_test.to(device)
Y_test = Y_test.to(device)
model = MultitaskGPModel(Z_train, Y_train, likelihood, car.n_states).to(device)
likelihood = likelihood.to(device)

##### Train the GP model #####
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
mse_list = []
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(Z_train)
    loss = -mll(output, Y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f    noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
    mse_list.append(loss.item())
    
# begin simulation of real life car
# Set into eval mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
    predictions = likelihood(model(Z_test))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    #print("predictions are ", predictions)
    #print("mean is ", mean)
    
    error = torch.pow(mean - Y_test, 2)
    print("error is ", error)
    
    mse = torch.mean(torch.pow(mean - Y_test, 2))
    print("mean squared error is ", mse)
    
#plot the errors
fig,ax = plt.subplots(3,1, figsize=(10,5))
ax[0].plot(Y_test[:,0].cpu().numpy(), label='True')
ax[0].plot(mean[:,0].cpu().numpy(), label='Predicted')
#plot confidence interval
ax[0].fill_between(np.arange(0,100), lower[:,0].cpu().numpy(), upper[:,0].cpu().numpy(), alpha=0.5)
ax[0].set_title('X Position')

ax[1].plot(Y_test[:,1].cpu().numpy(), label='True')
ax[1].plot(mean[:,1].cpu().numpy(), label='Predicted')
ax[1].fill_between(np.arange(0,100), lower[:,1].cpu().numpy(), upper[:,1].cpu().numpy(), alpha=0.5)
ax[1].set_title('Y Position')

ax[2].plot(Y_test[:,2].cpu().numpy(), label='True')
ax[2].plot(mean[:,2].cpu().numpy(), label='Predicted')
ax[2].fill_between(np.arange(0,100), lower[:,2].cpu().numpy(), upper[:,2].cpu().numpy(), alpha=0.5)
ax[2].set_title('Psi Angle')

plt.legend()

    
plt.show()