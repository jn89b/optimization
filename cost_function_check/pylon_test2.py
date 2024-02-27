import numpy as np
import matplotlib.pyplot as plt

"""
h = v^2/g
sin(phi) = h/r

#set equation for h
"""

## Let's try constraining our effector
r_min = 20
r_max = 50

# velocity bounds
v_min = 16
v_max = 30

# roll bounds
min_roll_deg = 1
max_roll_deg = 60

optimal_values = []

## This is just a pareto front sweep of your velocity and roll
for v in range(v_min, v_max+1):
    for phi in range(min_roll_deg, max_roll_deg):
        r = (v**2)/(9.81*np.sin(np.deg2rad(phi)))
        h = (v**2)/(9.81)   
        if r_min <= r <= r_max:
            optimal_values.append([v, phi, r, h])

for i in range(len(optimal_values)):
    print("v: ", optimal_values[i][0], "phi: ", 
          optimal_values[i][1], "r: ", 
          optimal_values[i][2], "h: ", optimal_values[i][3])
    

optimal_values = np.array(optimal_values)

fig,ax = plt.subplots(1,1, figsize=(10,10))
#plot v and phi with r as the color
#set color bar from min r to max r
color = np.linspace(min(optimal_values[:,2]), max(optimal_values[:,2]), len(optimal_values))
sc = ax.scatter(optimal_values[:,0], optimal_values[:,1], c=color, cmap='viridis')
#show color bar
buffer = 5
ax.set_xlim([v_min-buffer, v_max+buffer])
ax.set_ylim([min_roll_deg-buffer, max_roll_deg+buffer])
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('Roll (deg)')
cbar = plt.colorbar(sc)
cbar.set_label('Range (m)')
ax.set_title('Sweep of Velocity bounds and Roll bounds with effector range')

fig, ax = plt.subplots(1,1, figsize=(10,10))
color = np.linspace(min(optimal_values[:,3]), max(optimal_values[:,3]), len(optimal_values))
sc = ax.scatter(optimal_values[:,0], optimal_values[:,1], c=color, cmap='viridis')
buffer = 5
ax.set_xlim([v_min-buffer, v_max+buffer])
ax.set_ylim([min_roll_deg-buffer, max_roll_deg+buffer])
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('Roll (deg)')
cbar = plt.colorbar(sc)
cbar.set_label('Height (m)')
ax.set_title('Sweep of Velocity bounds and Roll bounds with effector height')

fig, ax = plt.subplots(1,1, figsize=(10,10))
color = np.linspace(min(optimal_values[:,1]), max(optimal_values[:,1]), len(optimal_values))
sc = ax.scatter(optimal_values[:,2], optimal_values[:,3], c=color, cmap='viridis')
ax.set_xlabel('Height (m)')
ax.set_ylabel('Range (m)')
cbar = plt.colorbar(sc)
cbar.set_label('Roll (deg)')
# ax.set_title('Sweep of Velocity bounds and Roll bounds with Roll

#plot a 3d image of the position fo the aircraft and its roll angle
fig, ax = plt.subplots(1,1, figsize=(10,10), subplot_kw={'projection': '3d'})
r_slant = optimal_values[:,2]
phi_rad = np.deg2rad(optimal_values[:,1])
x = r_slant
y = r_slant
h = optimal_values[:,3]
color = np.linspace(min(optimal_values[:,1]), max(optimal_values[:,1]), len(optimal_values))
sc = ax.scatter(x, y, h, c=color, cmap='viridis')    
#draw a quiver that points in the direction of the roll
u = -np.cos(phi_rad)
v = -np.sin(phi_rad)
w = 0
# ax.quiver(x, y, h, u, v, w, length=5.0, normalize=True)
cbar = plt.colorbar(sc)
cbar.set_label('Roll (deg)')
ax.set_xlim([0, max(x)+buffer])
ax.set_ylim([0, max(y)+buffer])
ax.set_zlim([0, max(h)+buffer])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


buffer = 1
color = np.linspace(min(h), max(h), len(optimal_values))
fig, ax = plt.subplots(1,1, figsize=(10,10), subplot_kw={'projection': '3d'})
sc = ax.scatter(optimal_values[:,0], np.rad2deg(phi_rad), r_slant, c=color, cmap='viridis')
cbar = plt.colorbar(sc)
cbar.set_label('Height (m)')
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('Roll (deg)')
ax.set_zlabel('Range (m)')
# ax.set_xlim([v_min-buffer, v_max+buffer])
# ax.set_ylim([np.deg2rad(min_roll_deg)-buffer, np.deg2rad(max_roll_deg)+buffer])
# ax.set_zlim([0, max(r_slant)+buffer])


plt.show()



    


        