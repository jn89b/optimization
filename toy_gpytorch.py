import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import pyplot as plt

def franke(X, Y):
    term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
    term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
    term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
    term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

    f = term1 + term2 + term3 - term4
    dfx = -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 + \
          -2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
    dfy = -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 + \
          -2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4

    return f, dfx, dfy, f

xv, yv = torch.meshgrid(torch.linspace(0, 1, 10), torch.linspace(0, 1, 10), indexing="ij")
#create a vector of ones and then concatenate the x and y values
zv = torch.ones(100)
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1)),
    dim=1
)

# train_y = torch.stack([
#     torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
#     torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
#     1.0 - train_x + torch.randn(train_x.size()) * 0.2,
# ], -1)


xv, yv = torch.meshgrid(torch.linspace(0, 1, 10), torch.linspace(0, 1, 10), indexing="ij")
#create a vector of ones and then concatenate the x and y values
zv = torch.ones(100)
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1)),
    dim=1
)

f, dfx, dfy, some_val = franke(train_x[:, 0], train_x[:, 1])
train_y = torch.stack([f, dfx, dfy, f], -1).squeeze(1)
# train_x = torch.linspace(0, 1, 100)


NUM_TASKS = train_y.shape[1]


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
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


print("train_x is ", train_x.shape)
print("train_y is ", train_y.shape)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=NUM_TASKS)
model = MultitaskGPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()


# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)


# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    
    
# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
fig, ax = plt.subplots(2, 3, figsize=(14, 10))

# Test points
n1, n2 = 50, 50
xv, yv = torch.meshgrid(torch.linspace(0, 1, n1), torch.linspace(0, 1, n2), indexing="ij")
f, dfx, dfy, f = franke(xv, yv)

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
    test_x = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
    predictions = likelihood(model(test_x))
    mean = predictions.mean

extent = (xv.min(), xv.max(), yv.max(), yv.min())
ax[0, 0].imshow(f, extent=extent, cmap=cm.jet)
ax[0, 0].set_title('True values')
ax[0, 1].imshow(dfx, extent=extent, cmap=cm.jet)
ax[0, 1].set_title('True x-derivatives')
ax[0, 2].imshow(dfy, extent=extent, cmap=cm.jet)
ax[0, 2].set_title('True y-derivatives')

ax[1, 0].imshow(mean[:, 0].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
ax[1, 0].set_title('Predicted values')
ax[1, 1].imshow(mean[:, 1].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
ax[1, 1].set_title('Predicted x-derivatives')
ax[1, 2].imshow(mean[:, 2].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
ax[1, 2].set_title('Predicted y-derivatives')
plt.show()