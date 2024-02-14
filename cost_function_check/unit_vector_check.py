import numpy as np
"""
Need to figure out which direction I should have the drive by at 
Compute the unit vector direction of the ego vehicle
Compute the los unit vector direction of the target
Compute the cross product of los to ego 


If user sets a goal position and an omnidirectinal effector, we want to offset 
the location of the omnidirectional effector be to the left or right 
of the goal position depending on direction of the goal position.

If the goal position is to the left of the ego vehicle, then the omnidirectional effector
- 
"""

# def drive_by_direction(goal_position):
#     pass


def find_driveby_direction(goal_position:np.ndarray, current_position:np.ndarray, 
                            heading_rad:float, effector_range:float):
    """
    Finds the lateral offset directions of the omnidirectional effector
    """    
    # los_unit_vector = np.array([np.cos(los_target), np.sin(los_target)])
    ego_unit_vector = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    
    #swap the direction sign to get the normal vector
    drive_by_vector_one = np.array([ego_unit_vector[1], -ego_unit_vector[0]])
    drive_by_vector_two = np.array([-ego_unit_vector[1], ego_unit_vector[0]])
    
    #pick the one closer to the current position
    distance_one = np.linalg.norm(current_position - (goal_position + drive_by_vector_one))
    distance_two = np.linalg.norm(current_position - (goal_position + drive_by_vector_two))
    
    if distance_one < distance_two:
        drive_by_vector = drive_by_vector_one
    else:
        drive_by_vector = drive_by_vector_two
        
    #apply to goal position
    drive_by_position = goal_position + drive_by_vector
    
    return drive_by_position

goal_position = np.array([10, 10])
ego_position = np.array([5, 5])
ego_heading = np.deg2rad(90)

#compute the unit vector of the ego vehicle
ego_unit_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])   

#compute the unit vector of the target 
los = np.arctan2(goal_position[1] - ego_position[1], goal_position[0] - ego_position[0])
los_unit_vector = np.array([np.cos(los), np.sin(los)])

#from ego to los
delta_vector = los_unit_vector - ego_unit_vector
print("Ego Unit Vector: ", ego_unit_vector)
print("LOS Unit Vector: ", los_unit_vector)
print("Delta Vector: ", delta_vector)
difference = np.dot(ego_unit_vector, los_unit_vector)

#flip the delta_vector and put it on the goal position
flipped_delta = -delta_vector
#multiply by R 
effector_range = 5
#flipped_delta = flipped_delta * 

drive_by_position = find_driveby_direction(goal_position, ego_position, ego_heading, effector_range)
print("Drive By Position: ", drive_by_position)

#cross product of the two vectors
cross_product = np.cross(ego_unit_vector, los_unit_vector)

# if cross product is positive, then the target is to the left of the ego vehicle
if cross_product > 0:
    print("Target is to the left of the ego vehicle")
    
    #if its to the left of the ego vehicle, then the ego vehicle should drive by to the right
else:
    print("Target is to the right of the ego vehicle")
# if cross product is negative, then the target is to the right of the ego vehicle

print("Cross Product: ", cross_product)

#plot the vectors 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot 2d 
fig, ax = plt.subplots()
ax.quiver(ego_position[0], ego_position[1], ego_unit_vector[0], ego_unit_vector[1], 
          angles='xy', scale_units='xy', scale=1, color='b', label='Ego Vector')

ax.quiver(ego_position[0], ego_position[1], los_unit_vector[0], los_unit_vector[1], 
          angles='xy', scale_units='xy', scale=1, color='r', label='LOS Vector')

ax.scatter(goal_position[0], goal_position[1], color='g', label='Goal Position')
ax.scatter(ego_position[0], ego_position[1], color='b', label='Ego Position')

#plot the delta vector
ax.quiver(ego_position[0], ego_position[1], delta_vector[0], delta_vector[1], 
          angles='xy', scale_units='xy', scale=1, color='k', label='Delta Vector')

#plot the flipped delta vector
ax.quiver(goal_position[0], goal_position[1], flipped_delta[0], flipped_delta[1], 
          angles='xy', scale_units='xy', scale=1, color='c', label='Flipped Delta Vector')

#plot the drive by position
ax.scatter(drive_by_position[0], drive_by_position[1], color='m', label='Drive By Position')

ax.set_xlim(-10, 20)
ax.set_ylim(-10, 20)

ax.legend()
plt.show()