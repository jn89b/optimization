import gpytorch




class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_states:int):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        # num task is the number of outputs from your model (n_x)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_states
        )
        
        self.base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_kernel, num_tasks=num_states, rank=1
        )
        # self.mean_module = gpytorch.means.MultitaskMean(
        #     gpytorch.means.ZeroMean(), num_tasks=num_states
        # )
        
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_states, rank=1
        )
        
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.MaternKernel(), num_tasks=num_states, rank=1
        # )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) 
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
