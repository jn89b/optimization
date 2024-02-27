import numpy as np
import matplotlib.pyplot as plt

v_min = 10
v_max = 30
min_roll = np.deg2rad(0.1)
max_roll = np.deg2rad(45)
g = 9.81
r_effector = 10


### Constraint this to the effector range and the min and max rolls of the aircraft
min_h = r_effector*np.sin(min_roll)
max_h = r_effector*np.sin(max_roll)

print("min_h: ", min_h)
print("max_h: ", max_h)

### Compute the h value for the min and max velocities
min_velocity = np.sqrt(2*g*min_h)
max_velocity = np.sqrt(2*g*max_h)

## we want to see if the min and max velocities are within the bounds if not to
## update the bound to match our v_min
print("min_velocity: ", min_velocity)
print("max_velocity: ", max_velocity)

if min_velocity < v_min:
    print("min_velocity is less than v_min")
    # if it is less then we need to constraint it to v_min
    for v in range(v_min, 10):
        min_h = (v_min**2)/(2*g)
        print("h: ", min_h)
        min_roll_ref = np.arcsin(min_h/r_effector)
        print("min_roll_ref: ", np.rad2deg(min_roll_ref))
    
if max_velocity > v_max:
    print("max_velocity is greater than v_max")

fig,ax = plt.subplots(1,1)


