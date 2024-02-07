# https://github.com/helgeanl/GP-MPC/blob/master/examples/tank_example.py
import casadi as ca
import numpy as np
import torch 
import gpytorch
import casadi as ca
import pyDOE

from gaussian_processes.gp_class import GP

class ToyCar():
    """
    Toy Car Example 
    
    3 States: 
    [x, y, psi]
     
     2 Inputs:
     [v, psi_rate]
    
    """
    def __init__(self):
        self.define_states()
        self.define_controls()
        
    def define_states(self) -> None:
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.psi = ca.SX.sym('psi')
        
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.psi
        )
        #column vector of 3 x 1
        self.n_states = self.states.size()[0] #is a column vector 
        
    def define_controls(self) -> None:
        self.v_cmd = ca.SX.sym('v_cmd')
        self.psi_cmd = ca.SX.sym('psi_cmd')
        
        self.controls = ca.vertcat(
            self.v_cmd,
            self.psi_cmd
        )
        #column vector of 2 x 1
        self.n_controls = self.controls.size()[0] 
        
    def set_state_space(self) -> None:
        #this is where I do the dynamics for state space
        self.x_dot = self.v_cmd * ca.cos(self.psi)
        self.y_dot = self.v_cmd * ca.sin(self.psi)
        self.psi_dot = self.psi_cmd
        
        self.z_dot = ca.vertcat(
            self.x_dot, self.y_dot, self.psi_dot    
        )
        
        #ODE right hand side function
        self.function = ca.Function('dynamics', 
                        [self.states, self.controls],
                        [self.z_dot]
                    ) 
        
    def rk45(self, x, u, dt, use_numeric:bool=True):
        """Runge-Kutta 4th order integration"""
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
        
    def generate_training_data(self, N:int, dt:float, constraint_params:dict, 
                               noise:bool=True):
        """ Generate training data using latin hypercube design
        # Arguments:
            N:   Number of data points to be generated
            uub: Upper input range (Nu,1)
            ulb: Lower input range (Nu,1)
            xub: Upper state range (Ny,1)
            xlb: Lower state range (Ny,1)

        # Returns:
            Z: Matrix (N, Nx + Nu) with state x and inputs u at each row
            Y: Matrix (N, Nx) where each row is the state x at time t+dt,
                with the input from the same row in Z at time t.
        """
        # Make sure boundary vectors are numpy arrays
        N = int(N)
        
        noise_val = 1e-3
        R = np.eye(self.n_states) * noise_val
        
        #check if the constraint parameters are a numpy array
        for key, value in constraint_params.items():
            if not isinstance(value, np.ndarray):
                constraint_params[key] = np.array(value)
        
        xlb = constraint_params["lower_state_bounds"]
        xub = constraint_params["upper_state_bounds"]
        ulb = constraint_params["lower_input_bounds"]
        uub = constraint_params["upper_input_bounds"]
        
        #state outputs with noise
        Y = np.zeros((N, self.n_states))
        
        # create control input design using a latin hypercube
        # latin hypercube dsign for unit cube [0,1]
        if self.n_controls > 0:
            U = pyDOE.lhs(self.n_controls, samples=N, criterion='maximin')
            for k in range(N):
                U[k] = U[k]*(uub - ulb) + ulb
        else:
            U = np.zeros((N, self.n_controls))
            
        #create state input design using latin hypercube doing the same thing
        X = pyDOE.lhs(self.n_states, samples=N, criterion='maximin')
        
        #scale the state inputs to the actual state space
        for k in range(N):
            X[k] = X[k]*(xub - xlb) + xlb
            
        num_parameters = 0
        parameters = pyDOE.lhs(num_parameters, samples=N)
        
        for i in range(N):
            if self.n_controls > 0:
                Y[i,:] = self.rk45(X[i], U[i], dt)
            else:
                Y[i,:] = self.rk45(X[i], np.array([]), dt)
                
            if noise:
                Y[i,:] += np.random.multivariate_normal(np.zeros(self.n_states), R)
            
        #concatenate the state and control inputs
        if self.n_controls > 0:
            Z = np.hstack((X, U))
        else:
            Z = X
        return Z, Y            


### System parameters for car
dt = 0.1 
Nx = 3 # Number of states [x, y, psi]
Nu = 2 # Number of inputs [v, psi_rate]

## Lower and upper bounds for states and inputs
xlb = np.array([-10, -10, -np.pi])
xub = np.array([10, 10, np.pi])
ulb = np.array([-1, -np.pi/4])
uub = np.array([1, np.pi/4])

constraint_params = {
    "lower_state_bounds": xlb,
    "upper_state_bounds": xub,
    "lower_input_bounds": ulb,
    "upper_input_bounds": uub,
}

num_samples = 100
num_tests   = 100

car = ToyCar()
car.set_state_space()
## We want to train the gp to find the residuals of our system
Z, Y = car.generate_training_data(num_samples, dt, constraint_params)

Z_test, Y_test = car.generate_training_data(num_tests, dt, constraint_params)


#convert Z and Y to torch tensors
Z = torch.tensor(Z).float()
Y = torch.tensor(Y).float()

Z_test = torch.tensor(Z_test).float()
Y_test = torch.tensor(Y_test).float()

print(Z.shape)
print(Y.shape)

NUM_TASKS = Y.shape[1]

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        #num task is the number of outputs from your model (n_x)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=NUM_TASKS
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=NUM_TASKS, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


#check if the model is on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)
Z = Z.to(device)
Y = Y.to(device)
Z_test = Z_test.to(device)
Y_test = Y_test.to(device)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=NUM_TASKS)
model = MultitaskGPModel(Z, Y, likelihood).to(device)

model = model.to(device)
likelihood = likelihood.to(device)

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)


# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
mse_list = []
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(Z)
    loss = -mll(output, Y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    mse_list.append(loss.item())
    
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

# Make predictions
test_mse = []
for e in error:
    mse_val = torch.mean(e)
    #convert to cpu
    test_mse.append(mse_val.cpu().numpy())

mean = mean.cpu().numpy()
lower = lower.cpu().numpy()
upper = upper.cpu().numpy()


x_state = mean[:, 0]
y_state = mean[:, 1]
psi_state = mean[:, 2]

x_state_test = Y_test[:, 0].cpu().numpy()
y_state_test = Y_test[:, 1].cpu().numpy()
psi_state_test = Y_test[:, 2].cpu().numpy()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1, figsize=(14, 10))
t = np.arange(0, num_tests, 1)
ax[0].plot(t, x_state, label='Predicted')  
ax[0].scatter(t, x_state_test, label='True', color='r')
ax[0].fill_between(np.arange(len(x_state)), lower[:, 0], upper[:, 0], alpha=0.5)

ax[1].plot(t, y_state, label='Predicted')
ax[1].scatter(t, y_state_test, label='True', color='r')
ax[1].fill_between(np.arange(len(y_state)), lower[:, 1], upper[:, 1], alpha=0.5)

ax[2].plot(t, psi_state, label='Predicted')
ax[2].scatter(t, psi_state_test, label='True', color='r')
ax[2].fill_between(np.arange(len(psi_state)), lower[:, 2], upper[:, 2], alpha=0.5)

ax[0].set_title('X State')
ax[1].set_title('Y State')
ax[2].set_title('Psi State')

for a in ax:
    a.legend()
    a.grid(True)

fig,ax = plt.subplots(1, 1, figsize=(14, 10))
ax.plot(mse_list, label='Training MSE')
ax.plot(test_mse, label='Test MSE')

ax.set_title('Mean Squared Error')
ax.set_xlabel('Iterations')
ax.set_ylabel('MSE')    
ax.legend()    
plt.show()

