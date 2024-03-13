import time 
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from scipy.spatial import distance


def does_right_side_have_more_obstacles(ego_heading:float,
                                    danger_zones:np.ndarray, right_array:np.ndarray,
                                    left_array:np.ndarray) -> bool:
    """
    Given position and heading we will determine which side has more obstacles by 
    computing the cross product of the ego_heading and the vector to the obstacle
    We will then sum up the right_array and return a boolean value 
    it will return true if the right side has more obstacles and 
    false if the left side has more obstacles 
    """
    ego_unit_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    for i in range(len(danger_zones)):
        #compute the vector to the obstacle
        obs_position = danger_zones[i][:2]
        cross_product = np.cross(ego_unit_vector, obs_position)
        
        #if cross product is positive then obstacle is to the left 
        if cross_product > 0:
            left_array[i] = 1
        #if cross product is negative then obstacle is to the right
        else:
            right_array[i] = 1 
    
    sum_right = np.sum(right_array)
    sum_left = np.sum(left_array)
    
    if sum_right > sum_left:
        return True
    else:
        return False

# @jit(nopython=True)
def find_driveby_direction(goal_position:np.ndarray, current_position:np.ndarray, 
                            heading_rad:float,
                            obs_radius:float,  
                            robot_radius:float, 
                            consider_obstacles:bool=False, 
                            danger_zones:np.ndarray=None) -> np.ndarray:
    """
    Finds the lateral offset directions of the omnidirectional effector
    """    
    
    if goal_position.shape[0] != 2:
        goal_position = goal_position[:2]
    if current_position.shape[0] != 2:
        current_position = current_position[:2]
    
    
    range_total = obs_radius + robot_radius
    
    ego_unit_vector = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    
    #swap the direction sign to get the normal vector
    drive_by_vector_one = np.array([ego_unit_vector[1], -ego_unit_vector[0]])
    drive_by_vector_two = np.array([-ego_unit_vector[1], ego_unit_vector[0]])
    
    drive_by_vector_one = drive_by_vector_one * range_total
    drive_by_vector_two = drive_by_vector_two * range_total
    
    #pick the one closer to the current position
    distance_one = np.linalg.norm(current_position - (goal_position + drive_by_vector_one))
    distance_two = np.linalg.norm(current_position - (goal_position + drive_by_vector_two))
    
    
    if consider_obstacles:
        #make a decision baased on the obstacles
        is_right = does_right_side_have_more_obstacles(heading_rad, danger_zones,
                                                         np.zeros(len(danger_zones)),
                                                         np.zeros(len(danger_zones)))

        #
        cross_product = np.cross(ego_unit_vector, drive_by_vector_one)
        if cross_product > 0:
            #if the cross product is positive then the right vector is the left vector
            left_vector = drive_by_vector_one
            right_vector = drive_by_vector_two
        else:
            left_vector = drive_by_vector_two
            right_vector = drive_by_vector_one
        
        if is_right:
            #if right side has more obstacles then pick the left vector
            #figure out which vector is the left vector
            drive_by_vector = left_vector
        else:
            drive_by_vector = right_vector
            
    else:  
        if distance_one < distance_two:
            drive_by_vector = drive_by_vector_two
        else:
            drive_by_vector = drive_by_vector_one
                
        
    #apply to goal position
    drive_by_position = goal_position + drive_by_vector
    
    return drive_by_position


def knn_obstacles(obs:np.ndarray, ego_pos:np.ndarray, 
                  K:int=3, use_2d:bool=False):
    if use_2d:
        ego_position = ego_pos[:2]
        obstacles = obs[:,:2]
    
    nearest_indices = distance.cdist([ego_position], obstacles).argsort()[:, :K]
    nearest_obstacles = obs[nearest_indices]
    return nearest_obstacles[0], nearest_indices



def find_inline_obstacles(ego_unit_vector:np.ndarray, obstacles:np.ndarray, 
                          ego_position:np.ndarray, 
                          dot_product_threshold:float=0.0, 
                          use_2d:bool=False) -> tuple:
    '''
    compute the obstacles that are inline with the ego
    '''
    #check size of ego_position
    if use_2d:
        if ego_position.shape[0] != 2:
            ego_position = ego_position[:2]
        current_obstacles = obstacles[:,:2]
        
    inline_obstacles = []
    for i, obs in enumerate(current_obstacles):
        los_vector = obs - ego_position
        los_vector /= np.linalg.norm(los_vector)
        dot_product = np.dot(ego_unit_vector, los_vector)
        if dot_product > dot_product_threshold:
            inline_obstacles.append((obstacles[i],dot_product))
            
    return inline_obstacles

def find_danger_zones(obstacles:np.ndarray, 
                      ego_position:np.ndarray, 
                      min_radius_turn:float, 
                      distance_buffer:float=10.0, 
                      use_2d:bool=False) -> np.ndarray:
    '''
    compute the obstacles that are inline with the ego
    '''
    #check size of ego_position
    if use_2d:
        if ego_position.shape[0] != 2:
            ego_position = ego_position[:2]
        curr_obstacles = obstacles[:,:2]
    
    danger_zones = []
    for i, obs, in enumerate(obstacles):
        obs_position = obs[:2]
        obs_radius = obs[-1]
        print("obs_position:", obs_position)

        # print("dot:", dot)
        distance_to_obstacle = np.linalg.norm(obs_position - ego_position)
        print("distance_to_obstacle:", distance_to_obstacle)
        delta_r = distance_to_obstacle +obs_radius - min_radius_turn
        if delta_r >= distance_buffer:
            #compute the danger zone
            danger_zones.append((obs,obstacles[i][1]))
            
    return danger_zones


current_position = np.array([10.0,10.0,0])
current_heading = np.deg2rad(45)
ego_unit_vector = np.array([np.cos(current_heading), np.sin(current_heading)])
current_velocity = 15
max_velocity = 30
min_velocity = 15
phi_max = np.deg2rad(45)
g = 9.81

r_thesholdhold = (min_velocity)**2/(g*np.tan(phi_max))
print("R Threshold:", r_thesholdhold)
random_seed = 42

np.random.seed(random_seed)

#generate random obstacles
num_obstacles = 10
# obs_x = np.random.uniform(-20, 100, num_obstacles)
# obs_y = np.random.uniform(20, 100, num_obstacles)
# obs_z = np.random.uniform(-10, 100, num_obstacles)
# obs_radii = np.random.uniform(3, 10, num_obstacles)

goal_x = 100 
goal_y  = 100
goal_z = 50

#generate a line of obstacles
obs_x = np.arange(95, 105, 1)
obs_y = np.ones(len(obs_x)) * goal_y
obs_z = np.ones(len(obs_x)) * goal_z
obs_radii = np.random.uniform(3, 10, len(obs_x))

obstacles = np.array([obs_x, obs_y, obs_z, obs_radii]).T

nearest_obstacles,nearest_indices  = knn_obstacles(
    obstacles, current_position, K=10, use_2d=True)

inline_obs = find_inline_obstacles(
    ego_unit_vector, 
    nearest_obstacles, current_position, use_2d=True)

danger_zones = find_danger_zones(
    inline_obs, current_position, r_thesholdhold, use_2d=True)
print("Danger Zones:", danger_zones)    

dot_product = [d[1] for d in inline_obs]
danger_zones = [d[0] for d in danger_zones]

max_dot_index = np.argmax(dot_product)
max_dot_obs = inline_obs[max_dot_index][0]
print("Max Dot Obs:", max_dot_obs)

driveby_position = find_driveby_direction(max_dot_obs[:2], current_position[:2], 
                                           current_heading,
                                             max_dot_obs[-1], 
                                             r_thesholdhold/2, 
                                             True, nearest_obstacles)
print("Driveby Position:", driveby_position)

#plot obstacles and direction vectors
fig, ax = plt.subplots(1,1, figsize=(10,10))

#draw circles
for obs in obstacles:
    ax.add_patch(plt.Circle(obs[:2], obs[-1], fill=False, color='r'))
        
#plot ego position vector
ax.quiver(current_position[0], current_position[1], 
          np.cos(current_heading), np.sin(current_heading), 
          color='b', scale=5)

for obs in danger_zones:
    ax.scatter(obs[0], obs[1], c='g', s=100)

#plot ego position vector
ax.quiver(current_position[0], current_position[1], 
          np.cos(current_heading), np.sin(current_heading), 
          color='b', scale=5)
#draw turn radius circle on the ego position offset by the heading
ax.add_patch(plt.Circle(current_position[:2], r_thesholdhold, fill=False, color='g'))

#plot the drive by position
ax.scatter(driveby_position[0], driveby_position[1], c='y', s=100)
plt.show()
