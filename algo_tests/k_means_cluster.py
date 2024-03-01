import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Example obstacles as coordinates
# obstacles = np.array([[1, 2], [2, 1], [5, 5], [5, 6], [8, 8], [9, 9]])
seed_number = 42
num_obstacles = 20

np.random.seed(seed_number)
obs_x = np.random.uniform(0, 150, num_obstacles)
obs_y = np.random.uniform(0, 150, num_obstacles)
obs_radii = np.random.uniform(1, 3, num_obstacles)
obstacles = np.array([obs_x, obs_y]).T

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