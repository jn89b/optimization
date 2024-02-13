import matplotlib.pyplot as plt
import numpy as np

# Example data
x_values = np.random.rand(100)  # Replace with your actual x values
y_values = np.random.rand(100)  # Replace with your actual y values
time_values = np.linspace(0, 10, 100)  # Time from 0 to 10 for example, replace with your actual time values

# Plot
scatter = plt.scatter(x_values, y_values, c=time_values, cmap='viridis')

# Add a color bar
plt.colorbar(scatter, label='Time')

# Label axes
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Scatter Points as a Function of Time')

plt.show()

