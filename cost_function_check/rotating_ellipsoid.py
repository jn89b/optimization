import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the radii of the ellipsoid
a, b, c = 4, 2, 1

# Define rotation matrices
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# Generate points on the ellipsoid surface
u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
x = a * np.cos(u) * np.sin(v)
y = b * np.sin(u) * np.sin(v)
z = c * np.cos(v)

# Prepare plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize plot
def init():
    ax.clear()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

# Animation update function
def update(frame):
    ax.clear()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    # Rotate the ellipsoid
    theta = np.radians(frame)
    Rx = rotation_matrix([1, 0, 0], theta)  # Rotation around x
    Ry = rotation_matrix([0, 1, 0], theta)  # Rotation around y
    Rz = rotation_matrix([0, 0, 1], theta)  # Rotation around z
    R = np.dot(Rx, np.dot(Ry, Rz))  # Combine rotations

    # Apply rotation
    x_rot, y_rot, z_rot = np.empty_like(x), np.empty_like(y), np.empty_like(z)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            [x_rot[i, j], y_rot[i, j], z_rot[i, j]] = np.dot(R, [x[i, j], y[i, j], z[i, j]])

    # Plot the rotated ellipsoid
    ax.plot_surface(x_rot, y_rot, z_rot, color='b', edgecolor='none')

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), init_func=init, blit=False)

plt.show()
