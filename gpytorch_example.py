import gpytorch
import casadi as ca
import numpy as np 
import pyDOE
import torch 

from gpytorch.mlls import SumMarginalLogLikelihood

# https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html

"""
https://docs.gpytorch.ai/en/v1.6.0/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_CUDA.html
""" 

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

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        #these are prior distributions for the mean and covariance
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # def log_prob(self, value:torch.Tensor) -> torch.Tensor:
    #     print("what is value: ", value)
    
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

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

num_samples = 15
num_tests   = 100
training_iteration = 50

car = ToyCar()
car.set_state_space()
Z, Y = car.generate_training_data(num_samples, dt, constraint_params)
Z_test, Y_test = car.generate_training_data(num_tests, dt, constraint_params)

#convert data to cuda
Z = torch.tensor(Z, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
Z_test = torch.tensor(Z_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

train_x1 = Z[:,0]
train_x2 = Z[:,1]
train_x3 = Z[:,2]
train_x4 = Z[:,3]
train_x5 = Z[:,4]

likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
likelihood3 = gpytorch.likelihoods.GaussianLikelihood()
likelihood4 = gpytorch.likelihoods.GaussianLikelihood()
likelihood5 = gpytorch.likelihoods.GaussianLikelihood()

model1 = MultitaskGPModel(train_x1, Y, likelihood1)
model2 = MultitaskGPModel(train_x2, Y, likelihood2)
model3 = MultitaskGPModel(train_x3, Y, likelihood3)
model4 = MultitaskGPModel(train_x4, Y, likelihood1)
model5 = MultitaskGPModel(train_x5, Y, likelihood2)

model = gpytorch.models.IndependentModelList(model1, model2, model3, model4, model5)
likelihood = gpytorch.likelihoods.LikelihoodList(likelihood1, likelihood2, likelihood3, likelihood4, likelihood5)
mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters


for i in range(training_iteration):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(*model.train_inputs)
    print("OUTPUT: ", output)
    # Calc loss and backprop gradients
    loss = -mll(output, model.train_targets)
    print("LOSS: ", loss)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iteration, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
    
