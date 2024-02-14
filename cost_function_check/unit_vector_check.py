import numpy as np
"""
Need to figure out which direction I should have the drive by at 
Compute the unit vector direction of the ego vehicle
Compute the los unit vector direction of the target
Compute the cross product of los to ego 
- 
"""

# def drive_by_direction(goal_position):
#     pass

goal_position = np.array([0, 10])

ego_position = np.array([0, 5])
ego_heading = np.deg2rad(45)

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

# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_aspect('equal')

ax.legend()
plt.show()