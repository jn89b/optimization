import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_inside_toroid(x:float, y:float, z:float, R:float, r:float) -> bool:
    return (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4*(R**2)*(x**2 + y**2) <= 0

# Parameters for the toroid
R = 10  # Major radius (distance from the center of the tube to the center of the torus)
r = 3   # Minor radius (radius of the tube)
num_points = 10
# Meshgrid of angles
theta = np.linspace(0, 2*np.pi, num_points)
phi = np.linspace(  0, 2*np.pi, num_points)
theta, phi = np.meshgrid(theta, phi)

# Parametric equations for the toroid
X = (R + r*np.cos(phi)) * np.cos(theta)
Y = (R + r*np.cos(phi)) * np.sin(theta)
Z = r*np.sin(phi)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, color='b', edgecolor='k')
ax.set_zlim(-15, 15)
ax.set_title('Toroid')

plt.show()