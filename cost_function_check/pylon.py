import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

max_roll = np.deg2rad(45)
min_roll = np.deg2rad(0.1)

minor_radius = 5
major_radius = 10
ideal_r = (major_radius - minor_radius)/2
max_range = major_radius + minor_radius

min_velocity = 15
max_velocity = 30
g = 9.81

min_h = max_range*np.sin(min_roll)
max_h = max_range*np.sin(max_roll)

n_vars = 4
n_traj = 1

opti = ca.Opti()
X = opti.variable(n_vars, n_traj)
h = X[0]
phi = X[1]
velocity = X[2]
radius = X[3]

tan_theta_cos_phi = ca.tan(phi)*ca.cos(phi)
tan_theta_cos_phi = ca.if_else(tan_theta_cos_phi == 0, 0.01, tan_theta_cos_phi)
tan_phi = ca.tan(phi)
tan_phi = ca.if_else(tan_phi == 0, 0.01, tan_phi)

r_slant_fun = (velocity**2/(g*tan_theta_cos_phi)) 

cost_fn = (ideal_r - r_slant_fun)
r_cost = ca.sumsqr(cost_fn)

opti.minimize(r_cost)
# Constraints
# opti.subject_to(opti.bounded(min_h, h, max_h)) 
opti.subject_to(opti.bounded(min_roll, phi, max_roll))
opti.subject_to(opti.bounded(min_velocity, velocity, max_velocity))

#set terminal constraints
# opti.subject_to(h >= ideal_h)
opti.subject_to(opti.bounded(minor_radius, radius, major_radius))

#set solver to ipopt and ma27
solver_options = {'ipopt': {'linear_solver': 'ma27'}}
opti.solver('ipopt', solver_options)
sol = opti.solve()

#get solution
print(sol.value(X))

#unpack solution
h_sol = sol.value(h)
phi_sol = sol.value(phi)
v_sol = sol.value(velocity)
r_sol = sol.value(radius)

print("h is", h_sol)
print("phi in degrees is", np.rad2deg(phi_sol))
print("v is", v_sol)
print("r is", r_sol)

print("min_h is", min_h)
print("max_h is", max_h)

print("cost function is", sol.value(r_slant_fun))
ground_radius = v_sol**2/(g*np.tan(phi_sol))
print("ground radius is", ground_radius)
print("actual h is", ground_radius*np.sin(phi_sol))

"""
Whats a good way to visualize this plot?
"""

pylon_x = 0
pylon_y = 0

distance_from_pylon = np.sqrt(ground_radius**2 + h_sol**2)
print("distance from pylon is", distance_from_pylon)

fig,ax = plt.subplots(1,1)
ax.scatter(ground_radius, h_sol, label='Aircraft Position')
#draw a quiver from the aircraft position based on the phi
x_direction = np.cos(phi_sol)
y_direction = np.sin(phi_sol)
ax.quiver(ground_radius, h_sol, x_direction, y_direction, angles='xy', 
          scale_units='xy', scale=1, color='r')

ax.scatter(0,0, label='Pylon Position')
ax.legend()
plt.show()

