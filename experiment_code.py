
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Gravity (m/s^2)
angle = 41  # Random launch angle
speed = 43  # Random speed (m/s)

# Convert to radians
theta = np.radians(angle)

# Time of flight, max height, and range
time_of_flight = (2 * speed * np.sin(theta)) / g
max_height = (speed**2 * np.sin(theta)**2) / (2 * g)
range_projectile = (speed**2 * np.sin(2 * theta)) / g

# Plot trajectory
t = np.linspace(0, time_of_flight, num=100)
x = speed * np.cos(theta) * t
y = speed * np.sin(theta) * t - 0.5 * g * t**2

plt.plot(x, y)
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Projectile Motion Simulation')
plt.show()
