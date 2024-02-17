import numpy as np 

from matplotlib import pyplot as plt

# go from 500 to 5
dtarget = np.arange(0, 250, 5)
print("dtarget: ", dtarget)
effector_range = 10

#compute the cost function
cost_fn_one = np.exp(-dtarget/effector_range)
velocity_penalty = 1 + (cost_fn_one/1)

velocity = 15 
total_cost = velocity_penalty * cost_fn_one
total_cost_2 = 25 * cost_fn_one

fig,ax = plt.subplots()
ax.plot(dtarget, cost_fn_one, 'r', label='1 - exp(d/effector_range)')
ax.set_xlabel('Distance to target')
ax.set_ylabel('Value')
ax.legend()
ax.grid()

fig,ax = plt.subplots()
ax.plot(dtarget, total_cost, 'b', label='Total cost at 15 m/s')
ax.plot(dtarget, total_cost_2, 'g', label='Total cost at 25 m/s')
ax.set_xlabel('Distance to target')
ax.set_ylabel('Value')

ax.legend()
ax.grid()

plt.show()


#plt.plot(dtarget, cost_fn_one, 'r', label='1 - exp(d/effector_range)')
#plt.plot(dtarget, cost_fn_one, 'r', label='$e^{-\Delta_r/{R_E}}$')
# plt.plot(dtarget, velocity_penalty, 'b', label='Velocity Penalty')
# plt.plot(dtarget, cost_fn_two, 'g', label='1 - d/effector_range')
# plt.xlabel('Distance to target')
# plt.ylabel('Value')

# plt.legend()
# plt.grid()
# # plt.tight_layout()
# plt.title('Effector with Range at 10 meters')
# plt.show()
#axis tight


#