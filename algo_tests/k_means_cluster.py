import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Example obstacles as coordinates
# obstacles = np.array([[1, 2], [2, 1], [5, 5], [5, 6], [8, 8], [9, 9]])
seed_number = 42
num_obstacles = 20

# np.random.seed(seed_number)
# obs_x = np.random.uniform(0, 150, num_obstacles)
# obs_y = np.random.uniform(0, 150, num_obstacles)
# obs_radii = np.random.uniform(1, 3, num_obstacles)
# obstacles = np.array([obs_x, obs_y]).T

goal_x = 100
goal_y = 100
goal_z = 100
robot_radius = 3.0
#generate a line of obstacles
obs_x = np.arange(goal_x-num_obstacles/2, goal_x+num_obstacles/2, 1)
obs_y = np.ones(len(obs_x)) * goal_y
obs_z = np.ones(len(obs_x)) * goal_z
obs_radii = np.random.uniform(3, 3.1, len(obs_x))
obstacles = np.array([obs_x, obs_y, obs_z, obs_radii]).T

# Number of clusters
K = 2  # Adjust based on your needs or use methods like the elbow method to determine

# Apply K-means clustering
kmeans = KMeans(n_clusters=K, random_state=0).fit(obstacles)

# Cluster centers (the combined obstacle positions)
centers = kmeans.cluster_centers_

print("Cluster Centers:", centers)

# Desired radius
#compute the radius of each cluster
radius = np.mean(cdist(obstacles, centers, 'euclidean'))

# Calculate distances of each obstacle to its cluster center
distances = cdist(obstacles, centers, 'euclidean')

print("distances:", distances)

#bundle the obstacles with their cluster centers
obstacles_with_centers = np.hstack((obstacles, kmeans.labels_.reshape(-1,1)))
#print which obstacles are in which cluster
for i in range(K):
    print("Obstacles in Cluster", i, ":", obstacles_with_centers[obstacles_with_centers[:,-1] == i])

# Check if each obstacle is within the radius of its cluster center
within_radius = distances <= radius

print("Obstacles within Radius of their Centers:", within_radius)

#plot the obstacles as circles
fig, ax = plt.subplots(1,1, figsize=(10,10))
#draw the circles
for i in range(len(centers)):
    ax.add_patch(plt.Circle(centers[i], radius, fill=False, color='r'))
    
#plot the obstacles
for i in range(len(obstacles)):
    ax.scatter(obstacles[:,0], obstacles[:,1], c='b')

plt.show()