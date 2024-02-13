import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def plot_gaussian(x_range, a, b, c):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = a * np.exp(-(x - b)**2 / (2 * c**2))
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Gaussian Function')
    plt.ylim(0, 1)  # Ensure y ranges from 0 to 1
    plt.xlim(x_range[0], x_range[1])  # Ensure x ranges from -1 to 1
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Function with x from -1 to 1 and y from 0 to 1')
    plt.legend()
    plt.show()

# Parameters
a = 1  # Peak height
b = 0  # Center of the peak
c = 1/3 # Adjusted for visual appeal within the given x range
# plot_gaussian((-1, 1), a, b, c)


x = ca.SX.sym('x')
gaussian_function = ca.Function('gaussian_function', [x], [a * ca.exp(-(x - b)**2 / (2 * c**2))],
                                ['x'], ['y'])

#test the function
dot_product = -1.0
y_val = gaussian_function(dot_product)
print('y_val:', y_val)