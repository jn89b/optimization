import numpy as np
import torch
import gpytorch
import casadi as ca # this will be done eventually

from matplotlib import pyplot as plt
from models.Plane import Plane
from gaussian_processes.gpytorch_interface import MultitaskGPModel
from controls.OptiMPC import OptiCasadi

import pandas as pd

"""
Notes to self:

States will be: 
    x,y,z,phi,theta,psi for now - 6 states
controls will be:
    roll_rate, pitch_rate, yaw_rate, airspeed - 4 controls

Z_train will then be N x 10

Y_train will be the output 
"""

def format_data(data:pd.DataFrame) -> np.ndarray:
    data['airspeed'] = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)
    Z = data[['x','y','z','phi','theta','psi','p','q','r','airspeed']].to_numpy()
    #drop the last row
    Z = Z[:-1,:]
    Y = data[['x','y','z','phi','theta','psi']].to_numpy()
    #drop the first row
    Y = Y[1:,:]
    return Z, Y


def get_residual_data(data:pd.DataFrame, model, num_states:int, 
                      num_controls:int) -> np.ndarray:
    data['airspeed'] = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)
    Z_data = data[['x','y','z','phi','theta','psi','p','q','r','airspeed']].to_numpy()
    Y = data[['x','y','z','phi','theta','psi']].to_numpy()
    
    num_states = 6
    x_states = Z_data[:, :num_states]
    u_states = Z_data[:, num_states:]
    
    Y_error = np.zeros((x_states.shape[0], num_states))
    nominal = np.zeros((x_states.shape[0], num_states))
    for i in range(0, x_states.shape[0]-1):
        x = x_states[i, :]
        u = u_states[i, :]
        x_dot = model.rk45(x, u, 1/10)
        nominal[i+1, :] = x_dot
        error = Y[i+1, :] - x_dot
        Y_error[i+1, :] = error
    
    return Z_data, Y_error, nominal

train_file_dir = 'flight_data/train/'
test_file_dir = 'flight_data/train/'
train_data = pd.read_csv(train_file_dir+'train_0.csv')
train_data['airspeed'] = np.sqrt(train_data['vx']**2 + train_data['vy']**2 + train_data['vz']**2)
Z_train = train_data[['x','y','z','phi','theta','psi','p','q','r','airspeed']].to_numpy()
#drop the last row
Y_train = train_data[['x','y','z','phi','theta','psi']].to_numpy()
#drop the first row
n_states = 6
n_controls = 4
x_states = Z_train[:, :n_states]
u_states = Z_train[:, n_states:]

plane = Plane()
plane.set_state_space()
Y_residuals_train = np.zeros((x_states.shape[0], n_states))
nominal_train = np.zeros((x_states.shape[0], n_states))
for i in range(0, x_states.shape[0]-1):
    x = x_states[i, :]
    u = u_states[i, :]
    x_dot = plane.rk45(x, u, 1/10)
    print("x", x)
    print("x dot", x_dot)
    nominal_train[i+1, :] = x_dot
    error =  Y_train[i+1, :] - x_dot
    Y_residuals_train[i+1, :] = error
    print('\n')
    
Y_train = Y_residuals_train
test_data = pd.read_csv(test_file_dir+'train_1.csv')
Z_test, Y_test, nominal_test = get_residual_data(test_data, plane, n_states, n_controls)

# convert to torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_train = torch.tensor(Z_train, device=device).float()
Y_train = torch.tensor(Y_train, device=device).float()
Z_test = torch.tensor(Z_test, device=device).float()
Y_test = torch.tensor(Y_test, device=device).float()


# initialize likelihood and model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_states)
likelihood.to(device)
model = MultitaskGPModel(Z_train, Y_train, likelihood, num_states=n_states).to(device)

## Train the model
model.train()
likelihood.train()


optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 150
old_loss = 0

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(Z_train)
    loss = -mll(output, Y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    delta_loss = old_loss - loss.item()
    old_loss = loss.item()
    
model.eval()
likelihood.eval()

#%% 
# Make predictions for error
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(Z_test))

    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()
    

#convert mean to numpy
mean = mean.cpu().numpy()
lower = lower.cpu().numpy()
upper = upper.cpu().numpy()
Y_test = Y_test.cpu().numpy()

#%% 
#plot the error
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
time_vector = np.arange(mean.shape[0])
labels = ['X', 'Y', 'Z']

x_error = Y_test[:, 0]
y_error = Y_test[:, 1]
z_error = Y_test[:, 2]

y_outputs = [x_error, y_error, z_error]

# ax[0].plot(mean[:, 0], label='Predicted', color='r')
# ax[0].scatter(time_vector, x_error, label='Actual')
# ax[0].fill_between(np.arange(mean.shape[0]), lower[:, 0], upper[:, 0], alpha=0.5)
# ax[0].set_title('X')
# ax[0].legend()

# ax[1].plot(mean[:, 1], label='Predicted', color='r')
# ax[1].scatter(time_vector, y_error, label='Actual')
# ax[1].fill_between(np.arange(mean.shape[0]), lower[:, 1], upper[:, 1], alpha=0.5)
# ax[1].set_title('Y')

# ax[2].plot(mean[:, 2], label='Predicted', color='r')
# ax[2].scatter(time_vector, z_error, label='Actual')
# ax[2].fill_between(np.arange(mean.shape[0]), lower[:, 2], upper[:, 2], alpha=0.5)
# ax[2].set_title('Z')

#roll pitch yaw
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
time_vector = np.arange(mean.shape[0])
labels = ['Phi', 'Theta', 'Psi']

phi_error = np.rad2deg(Y_test[:, 3])
theta_error = np.rad2deg(Y_test[:, 4])
psi_error = np.rad2deg(Y_test[:, 5])

y_outputs = [phi_error, theta_error, psi_error]

ax[0].plot(np.rad2deg(mean[:, 3]), label='Predicted', color='r')
ax[0].scatter(time_vector, phi_error, label='Actual')
ax[0].fill_between(np.arange(mean.shape[0]), np.rad2deg(lower[:, 3]), np.rad2deg(upper[:, 3]), alpha=0.5)
ax[0].set_title('Phi')

ax[1].plot(np.rad2deg(mean[:, 4]), label='Predicted', color='r')
ax[1].scatter(time_vector, theta_error, label='Actual')
ax[1].fill_between(np.arange(mean.shape[0]), np.rad2deg(lower[:, 4]), np.rad2deg(upper[:, 4]), alpha=0.5)
ax[1].set_title('Theta')

ax[2].plot(np.rad2deg(mean[:, 5]), label='Predicted', color='r')
ax[2].scatter(time_vector, psi_error, label='Actual')
ax[2].fill_between(np.arange(mean.shape[0]), np.rad2deg(lower[:, 5]), np.rad2deg(upper[:, 5]), alpha=0.5)
ax[2].set_title('Psi')
ax[2].legend()

#super plot title
fig.suptitle('Error Prediction')


#%%  Let's add the prediction to our nominal values now 
dynamics_corrected = nominal_test + mean
phi_corrected = np.rad2deg(dynamics_corrected[:, 3])
theta_corrected = np.rad2deg(dynamics_corrected[:, 4])
psi_corrected = np.rad2deg(dynamics_corrected[:, 5])

phi_nomial = np.rad2deg(nominal_test[:, 3])
theta_nomial = np.rad2deg(nominal_test[:, 4])
psi_nomial = np.rad2deg(nominal_test[:, 5])

true_dynamics = test_data[['phi', 'theta', 'psi']].to_numpy()
phi_true = np.rad2deg(true_dynamics[:, 0])
theta_true = np.rad2deg(true_dynamics[:, 1])
psi_true = np.rad2deg(true_dynamics[:, 2])

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
time_vector = np.arange(mean.shape[0])
labels = ['Phi', 'Theta', 'Psi']

#compute the mse for each
phi_corrected_mse = np.mean((phi_corrected - phi_true)**2)
theta_corrected_mse = np.mean((theta_corrected - theta_true)**2)
psi_corrected_mse = np.mean((psi_corrected - psi_true)**2)

phi_nominal_mse = np.mean((phi_nomial - phi_true)**2)
theta_nominal_mse = np.mean((theta_nomial - theta_true)**2)
psi_nominal_mse = np.mean((psi_nomial - psi_true)**2)


ax[0].plot(phi_corrected, label='Corrected GP', color='r')
ax[0].plot(phi_nomial, label='Nominal', color='b')
ax[0].scatter(time_vector, phi_true, label='True', color='g')
ax[0].set_title('Phi - MSE Corrected: %.3f, Nominal: %.3f' % (phi_corrected_mse, phi_nominal_mse))
ax[0].legend()

ax[1].plot(theta_corrected, label='Corrected GP', color='r')
ax[1].plot(theta_nomial, label='Nominal', color='b')
ax[1].scatter(time_vector, theta_true, label='True', color='g')
ax[1].set_title('Theta - MSE Corrected: %.3f, Nominal: %.3f' % (theta_corrected_mse, theta_nominal_mse))
ax[1].legend()

ax[2].plot(psi_corrected, label='Corrected GP', color='r')
ax[2].plot(psi_nomial, label='Nominal', color='b')
ax[2].scatter(time_vector, psi_true, label='True', color='g')
ax[2].set_title('Psi - MSE Corrected: %.3f, Nominal: %.3f' % (psi_corrected_mse, psi_nominal_mse))
ax[2].legend()
ax[2].set_ylim(-360, 360)

#super plot title
fig.suptitle('Dynamics Prediction with Test Data')
plt.show()
