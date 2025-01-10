import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Function to calculate curvature for a smooth path
def calculate_curvature(x, y):
    # First and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: k = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
    return curvature

# Function to calculate velocity based on curvature
def calculate_velocity(curvature, max_speed=10, min_speed=2):
    # Element-wise velocity calculation using numpy
    velocity = np.maximum(min_speed, np.minimum(max_speed, max_speed / (np.abs(curvature) + 0.1)))
    return velocity


# Define waypoints
waypoints = np.array([
    [0, 0],  # Start point
    [2, 5],
    [5, 7],
    [8, 5],
    [10, 0]  # End point
])

# Extract x and y coordinates
x_points, y_points = waypoints[:, 0], waypoints[:, 1]

# Create a cubic spline for smooth path
cs = CubicSpline(x_points, y_points)

# Generate smooth path points
x_smooth = np.linspace(min(x_points), max(x_points), 1000)
y_smooth = cs(x_smooth)

# Calculate curvature for the smooth path
curvature = calculate_curvature(x_smooth, y_smooth)

# Calculate velocity profile based on curvature
velocities = calculate_velocity(curvature)

# Plot the path and velocity profile
plt.figure(figsize=(12, 6))

# Plot the path
plt.subplot(1, 2, 1)
plt.plot(x_smooth, y_smooth, label="Smooth Path")
plt.scatter(x_points, y_points, color='red', zorder=5, label="Waypoints")
plt.title("Planned Path")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Plot the velocity profile
plt.subplot(1, 2, 2)
plt.plot(x_smooth, velocities, label="Velocity Profile", color='green')
plt.title("Velocity Profile Along Path")
plt.xlabel("Path Length (X)")
plt.ylabel("Velocity")
plt.legend()

plt.tight_layout()
plt.show()
