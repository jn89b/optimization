import casadi as ca
import numpy as np

r_min = 40
r_max = 50

v_min = 16
v_max = 30

phi_min = np.deg2rad(2)
phi_max = np.deg2rad(45) 

g = 9.81


n_states = 4

opti = ca.Opti()
r = opti.variable()
h = opti.variable()
v = opti.variable()
phi = opti.variable()

r_fn = v**2/(g*ca.sin(phi))

cost_fn = (r - r_fn)**2

opti.minimize(cost_fn)

opti.subject_to(opti.bounded(r_min, r, r_max))
opti.subject_to(opti.bounded(v_min, v, v_max))
opti.subject_to(opti.bounded(phi_min, phi, phi_max))
# opti.subject_to(h == v**2/(g))

#use ma27 linear solver
# solver_opts = {'linear_solver': 'ma27'}
solver_options = {'ipopt': {'linear_solver': 'ma27'}}

opti.solver('ipopt', solver_options)
sol = opti.solve()

print("r: ", sol.value(r))
print("h: ", sol.value(h))
print("v: ", sol.value(v))
print("phi: ", sol.value(phi))

