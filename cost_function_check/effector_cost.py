import numpy as np 

from matplotlib import pyplot as plt

# go from 500 to 5
dtarget = np.arange(0, 500, 5)
print("dtarget: ", dtarget)
effector_range = 50

#compute the cost function
cost_fn_one = np.exp(-dtarget/effector_range)
cost_fn_two = 1 - (effector_range/dtarget)

velocity_penalty = 1 + (cost_fn_one/1)

#plt.plot(dtarget, cost_fn_one, 'r', label='1 - exp(d/effector_range)')
plt.plot(dtarget, cost_fn_one, 'r', label='exp(d/effector_range)')
# plt.plot(dtarget, velocity_penalty, 'b', label='Velocity Penalty')
# plt.plot(dtarget, cost_fn_two, 'g', label='1 - d/effector_range')
plt.xlabel('Distance to target')
plt.ylabel('Cost')

plt.legend()
plt.grid()
plt.show()
