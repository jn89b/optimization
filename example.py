import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cylinder parameters
radius = 1
height = 2
center = [0, 0]  # x, y position of the center

# Create a grid of points
theta = np.linspace(0, 2*np.pi, 100)
z = np.linspace(0, height, 50)
theta, z = np.meshgrid(theta, z)
x = radius * np.cos(theta) + center[0]
y = radius * np.sin(theta) + center[1]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, color='b', alpha=0.5)

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
