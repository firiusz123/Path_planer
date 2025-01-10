import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Function to calculate curvature
def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
    return curvature

# Function to generate velocity profile based on curvature
def generate_velocity_profile(x, y, max_speed, max_accel, max_curvature):
    curvature = calculate_curvature(x, y)
    velocity_profile = np.zeros_like(curvature)
    
    for i in range(len(curvature)):
        velocity_profile[i] = max_speed / (1 + max_curvature * curvature[i])
        velocity_profile[i] = min(velocity_profile[i], max_speed)
    
    # Smooth the velocity profile to avoid jerk
    velocity_profile = np.convolve(velocity_profile, np.ones(5)/5, mode='same')
    
    return velocity_profile

# Function to plot the path and velocity profile
def plot_path_and_velocity(x, y, velocity_profile, selected_points):
    # Plot the path
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, y, label="Path", color='b')
    plt.scatter(selected_points[:, 0], selected_points[:, 1], color='red', label="Selected Points", zorder=5)
    plt.title("Optimized Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    
    # Plot the velocity profile without points
    plt.subplot(2, 1, 2)
    plt.plot(velocity_profile, label="Velocity Profile", color='g')
    plt.title("Velocity Profile Along Path")
    plt.xlabel("Path Segments")
    plt.ylabel("Velocity (units/s)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Step 1: Pick points using ginput
plt.figure(figsize=(10, 6))
plt.title("Select Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim([0, 640])
plt.ylim([0, 640])

selected_points = plt.ginput(n=-1, timeout=0)
print("Selected points:", selected_points)

# Extract the x and y coordinates from selected points
x_points, y_points = zip(*selected_points)

# Convert to numpy arrays for easy manipulation
selected_points = np.array(list(zip(x_points, y_points)))

# Step 2: Create the cubic spline for the path
x_points = np.array(x_points)
y_points = np.array(y_points)
cs = CubicSpline(x_points, y_points, bc_type='clamped')

# Generate a smooth path
x_smooth = np.linspace(min(x_points), max(x_points), 1000)
y_smooth = cs(x_smooth)

# Define parameters for velocity profiling
max_speed = 5  # maximum speed (units per second)
max_accel = 1  # maximum acceleration (units per second squared)
max_curvature = 10  # maximum allowable curvature for smooth turns

# Step 3: Generate the velocity profile based on curvature
velocity_profile = generate_velocity_profile(x_smooth, y_smooth, max_speed, max_accel, max_curvature)

# Step 4: Plot the path and velocity profile
plot_path_and_velocity(x_smooth, y_smooth, velocity_profile, selected_points)
