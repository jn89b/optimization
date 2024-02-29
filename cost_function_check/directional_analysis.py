import numpy as np
import matplotlib.pyplot as plt

def find_driveby_direction(goal_position:np.ndarray, current_position:np.ndarray, 
                            heading_rad:float,
                            robot_radius:float, radius_buffer:float):
    """
    Finds the lateral offset directions of the omnidirectional effector
    
    """    
    range_total = robot_radius + radius_buffer
    ego_unit_vector = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    #swap the direction sign to get the normal vector
    drive_by_vector_one = np.array([ego_unit_vector[1], -ego_unit_vector[0]])
    drive_by_vector_two = np.array([-ego_unit_vector[1], ego_unit_vector[0]])
    drive_by_vector_one = drive_by_vector_one * range_total
    drive_by_vector_two = drive_by_vector_two * range_total
    #pick the one closer to the current position
    distance_one = np.linalg.norm(current_position - (goal_position + drive_by_vector_one))
    distance_two = np.linalg.norm(current_position - (goal_position + drive_by_vector_two))


    if distance_one < distance_two:
        drive_by_vector = drive_by_vector_two
    else:
        drive_by_vector = drive_by_vector_one
            
    #apply to goal position
    drive_by_position = goal_position + drive_by_vector
    
    return drive_by_position

r_obs = 2 #m
v_min = 15 #m/s
phi = np.arctan2(v_min, r_obs*9.81)
print("phi: ", np.rad2deg(phi)) 

target_location = np.array([15, 15, 0])
ego_location = np.array([0, 0, 0])

"""
Idea is to continously check if within range and if phi manuever is possible 
If within range and and phi manuever is possible keep going in the current goal state 
If within range and phi manuever becomes impossible, then update the goal state to the drive by position


This belongs to the Decision drone 

set threshold_phi
So if within range:
    compute_phi_manuever 
    if phi_manuever+threshold_phi < phi_max:
        continue to goal state/engage 
    else:
        compute the drive by position/renenage
        
"""
