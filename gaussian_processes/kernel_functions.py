import numpy as np
import matplotlib.pyplot as plt
#import spatial distance function
from scipy.spatial.distance import cdist

# Define the exponentiated quadratic 
def exponentiated_quadratic(xa:np.ndarray, xb:np.ndarray):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    # covariance decreases with the square of the distance
    
    norm_distance = cdist(xa, xb, 'sqeuclidean')
    sq_norm = -0.5 * norm_distance
    return np.exp(sq_norm)

def compute_gaussian_process(X1, y1, X2, kernel_function):
    """
    Calculate the mean and covariance of the Gaussian process
    """
    # kernel of data points
    c11 = kernel_function(X, X)
    # kernel of observations vs to be predicted
    c12 = kernel_function(X, X2)
    #solve the linear system
    solved = np.linalg.solve(c11, c12)
    
    


#%% Plot the kernel function
# # samples from gaussian process
# num_samples = 100 # number of samples
# num_functions = 10 # number of functions

# X = np.expand_dims(np.linspace(-4, 4, num_samples), 1)
# COV_MATRIX = exponentiated_quadratic(X, X)  # Kernel of data points

# # Draw samples from the prior at our data points.
# # Assume a mean of 0 for simplicity
# ys = np.random.multivariate_normal(
#     mean=np.zeros(num_samples), cov=COV_MATRIX, 
#     size=num_functions)

# # plot the samples
# fig,ax = plt.subplots()
# for i in range(num_functions):
#     ax.plot(X, ys[i], alpha=0.7, linestyle='-', marker='o', markersize=3)
# ax.set_title('Samples from the GP prior')
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y=f(x)$')
# ax.set_title('Samples from the GP prior')
# plt.show()


#%% Predictions with Gaussian Process
