import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Define start and end points (for simplicity, in 2D)
start_point = np.array([0, 0])
end_point = np.array([10, 10])

# Define yaw angles for start and end direction vectors (in radians)
yaw_start = np.radians(0)  # 45 degrees converted to radians
yaw_end = np.radians(45)  # 

# Direction vectors defined by yaw angles
start_direction_yaw = np.array([np.cos(yaw_start), np.sin(yaw_start)])
end_direction_yaw = np.array([np.cos(yaw_end), np.sin(yaw_end)])

degree = 3
# Determine intermediate control points based on yaw-defined direction vectors
control_point_1_yaw = start_point + start_direction_yaw * 3  # Extend from start
control_point_2_yaw = end_point + end_direction_yaw * 3  # Extend towards end

# Control points including the start and end points based on yaw angles
control_points_yaw = np.array([start_point, control_point_1_yaw, control_point_2_yaw, end_point])

# Degree of the B-spline and knot vector

knots = np.linspace(0, 1, len(control_points_yaw) - degree + 1)
knots = np.append(np.zeros(degree), knots)
knots = np.append(knots, np.ones(degree))

# Create and evaluate the B-spline
spline_yaw = BSpline(knots, control_points_yaw, degree)
t = np.linspace(0, 1, 100)
spline_points_yaw = spline_yaw(t)

# Define the obstacle as a circle (center and radius)
obstacle_center = np.array([5, 5])
obstacle_radius = 2

# Function to check if a point is inside the obstacle (circle)
def is_point_inside_obstacle(point, center, radius):
    return np.linalg.norm(point - center) <= radius

# Check each point on the spline against the obstacle
hits_obstacle = any(is_point_inside_obstacle(point, obstacle_center, obstacle_radius) for point in spline_points_yaw)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(spline_points_yaw[:,0], spline_points_yaw[:,1], 'b', label='B-Spline Curve')
plt.plot(control_points_yaw[:,0], control_points_yaw[:,1], 'ro--', label='Control Points')
circle = plt.Circle(obstacle_center, obstacle_radius, color='g', alpha=0.5, label='Obstacle')
plt.gca().add_patch(circle)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('B-Spline Curve and Obstacle')
plt.grid(True)
plt.axis('equal')
plt.show()

# Output whether the spline hits the obstacle
print("Does the spline hit the obstacle?", hits_obstacle)
