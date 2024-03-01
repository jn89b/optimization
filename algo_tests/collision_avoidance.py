"""
## Obstacle avoidance 
- Use KNN to find nearest n obstacle 
- Check for the following
    - if heading towards the obstacle:
        - compute ego_unit_vector
        - compute los_from ego to los
        - dot_product:
            - if positive facing towards it
    - distance/manuever check
        - If radius of turn is greate
        - compute distance and compute phi to manuever:
            - $\phi_{computed} = v_{curr}^2/(gr)$
            - if $\phi_{computed} + \phi_{computed} >= \phi_{max}$
            - Then dangerous

    - If both these conditions hold true then I need to 
    set a reference point and depending on the 
    sign convention of the aircraft (left or right) 
    set the desired phi angle to the max or min phi angle  
    
"""

import numpy as np
import time 
from numba import jit
from scipy.spatial import distance


@jit(nopython=True)
def euclidean_distance_numba(vector1, vector2):
    '''calculate the euclidean distance,
    '''
    dist = np.linalg.norm(vector1-vector2)
    return dist


@jit(nopython=True)
def find_nearest_obstacle(obstacles:list, ego_position:np.ndarray, K:int=3, 
                          use_2d:bool=False) -> np.ndarray:
    """
    Find the nearest n obstacles from the ego_position
    """
    num_obstacles = len(obstacles)
    distances = np.zeros(num_obstacles)
    if use_2d:
        ego_position = ego_position[:2]
        obstacles = obstacles[:,:2]
        for i in range(num_obstacles):
            distances[i] = euclidean_distance_numba(ego_position, obstacles[i])
    else:
        for i in range(num_obstacles):
            distances[i] = euclidean_distance_numba(ego_position, obstacles[i])
        
    #sort the distances and return the indices of the nearest K obstacles
    nearest_indices = np.argsort(distances)[:K]
    near_obstacles = obstacles[nearest_indices]
    near_distance = distances[nearest_indices]

    return near_obstacles, near_distance, nearest_indices
    
@jit(nopython=True)
def compute_ego_unit_vector(heading:float) -> np.ndarray:
    '''compute the ego unit vector
    '''
    return np.array([np.cos(heading), np.sin(heading), 0])

@jit(nopython=True)
def compute_los_vector(ego_position:np.ndarray, nearest_obstacle:np.ndarray) -> np.ndarray:
    '''compute the line of sight vector
    '''
    return nearest_obstacle - ego_position

@jit(nopython=True)
def compute_dot_product(ego_unit_vector:np.ndarray, los_vector:np.ndarray) -> float:
    '''compute the dot product of the ego unit vector and the los vector
    '''
    return np.dot(ego_unit_vector, los_vector)

@jit(nopython=True)
def compute_obstacles_facing_ego(ego_unit_vector:np.ndarray, obstacles:np.ndarray, ego_position:np.ndarray) -> np.ndarray:
    '''compute the obstacles that are facing the ego
    '''
    facing_obstacles = np.zeros((0,3))
    for i in range(len(obstacles)):
        los_vector = compute_los_vector(ego_position, obstacles[i])
        los_vector /= np.linalg.norm(los_vector)
        dot_product = compute_dot_product(ego_unit_vector, los_vector)
        if dot_product > 0:
            facing_obstacles.append(obstacles[i])
    return np.array(facing_obstacles)

    
current_position = np.array([0,0,0])
current_heading = np.deg2rad(45)
random_seed = 42
np.random.seed(random_seed)

#generate random obstacles
num_obstacles = 100
obs_x = np.random.uniform(-10, 10, num_obstacles)
obs_y = np.random.uniform(-10, 10, num_obstacles)
obs_z = np.random.uniform(-10, 10, num_obstacles)
obstacles = np.array([obs_x, obs_y, obs_z]).T

#find the nearest obstacle
current_time = time.time()
nearest_obstacles, nearest_distance = find_nearest_obstacle(obstacles, current_position, K=3, use_2d=True)
end_time = time.time()
print("Nearest Obstacles: ", nearest_obstacles)
print("computed in: ", end_time - current_time, " seconds")
print("nearest_distance: ", nearest_distance)
#compare to baseline scipy.spatial.distance
current_time = time.time()
nearest_obstacles_scipy = distance.cdist([current_position], obstacles).argsort()[:, :3]
end_time = time.time()
print("nearest_obstacles using scipy: ", nearest_obstacles)
print("Comuputed using scipy: ", end_time - current_time, " seconds")
n_iter = 10

current_time = time.time()
for i in range(n_iter):
    start_time = time.time()
    nearest_obstacles, nearest_distance = find_nearest_obstacle(obstacles, current_position, K=3)
    print("current iteration: ", i, " time: ", time.time() - start_time)    


    #compute dot product of the ego unit vector and the los vector we want to see if the ego is facing the obstacle
    facing_obstacles = []
    start_time = time.time()
    for i in range(len(nearest_obstacles)):
        ego_unit_vector = np.array([np.cos(current_heading), np.sin(current_heading), 0])
        los_vector = nearest_obstacles[i] - current_position
        los_vector /= np.linalg.norm(los_vector)
        dot_product = np.dot(ego_unit_vector, los_vector)
        print("dot_product: ", dot_product)
        if dot_product > 0:
            facing_obstacles.append(nearest_obstacles[i])
    end_time = time.time()
    print("end_time: ", end_time - start_time)
    print("facing_obstacles: ", facing_obstacles)
        
    # #compute the obstacles that are facing the ego
    # start_time = time.time()
    # facing_obstacles = compute_obstacles_facing_ego(ego_unit_vector, obstacles, current_position)
    # print("facing_obstacles: ", facing_obstacles)
    
            
    