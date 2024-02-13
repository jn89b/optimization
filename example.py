import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_torus_and_point(R, r, point):
    # Generate torus
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    u, v = np.meshgrid(u, v)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    # Setup plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot torus
    ax.plot_surface(x, y, z, color='cyan', alpha=0.5)

    # Plot point
    px, py, pz = point
    ax.scatter(px, py, pz, color='red', s=100)  # Point in red

    # Annotations and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Torus with a Point')

    # Setting limits for better visualization
    ax.set_xlim([-R-2*r, R+2*r])
    ax.set_ylim([-R-2*r, R+2*r])
    ax.set_zlim([-2*r, 2*r])

    plt.show()

# Torus parameters
R = 5  # Major radius
r = 1  # Minor radius
point = (1, 1, 1)  # Point to check

plot_torus_and_point(R, r, point)
