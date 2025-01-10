import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# Function to calculate curvature
def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
    return curvature

# Function to apply curvature constraint to path (adjust path based on max curvature)
def apply_curvature_constraint(x, y, max_curvature):
    # Calculate the curvature of the original path
    curvature = calculate_curvature(x, y)
    
    # Apply the curvature constraint (limiting curvature)
    for i in range(1, len(curvature)):
        if curvature[i] > max_curvature/100:
            # Modify the path to smooth the transition (just an example approach)
            scale_factor = max_curvature / (curvature[i] * 100)
            x[i] = x[i-1] + (x[i] - x[i-1]) * scale_factor * 2  # Increased scaling factor
            y[i] = y[i-1] + (y[i] - y[i-1]) * scale_factor * 2  # Increased scaling factor
    
    return x, y

# Function to generate velocity profile based on curvature
def generate_velocity_profile(x, y, max_speed, min_speed, max_curvature, max_accel):
    curvature = calculate_curvature(x, y)
    velocity_profile = np.zeros_like(curvature)
    
    for i in range(len(curvature)):
        velocity_profile[i] = max_speed / (1 + max_curvature * curvature[i])
        velocity_profile[i] = max(min_speed, min(velocity_profile[i], max_speed))
    
    # Smooth the velocity profile to avoid jerk
    velocity_profile = np.convolve(velocity_profile, np.ones(5)/5, mode='same')
    
    return velocity_profile

# Function to plot the path and velocity profile
def plot_path_and_velocity(x, y, velocity_profile, selected_points, ax_path, ax_velocity):
    # Plot the path
    ax_path.clear()
    ax_path.plot(x, y, label="Path", color='b')
    ax_path.scatter(selected_points[:, 0], selected_points[:, 1], color='red', label="Selected Points", zorder=5)
    ax_path.set_title("Optimized Path")
    ax_path.set_xlabel("X")
    ax_path.set_ylabel("Y")
    ax_path.grid(True)
    ax_path.legend()

    # Plot the velocity profile
    ax_velocity.clear()
    ax_velocity.plot(velocity_profile, label="Velocity Profile", color='g')
    ax_velocity.set_title("Velocity Profile Along Path")
    ax_velocity.set_xlabel("Path Segments")
    ax_velocity.set_ylabel("Velocity (units/s)")
    ax_velocity.grid(True)
    ax_velocity.legend()

    plt.draw()

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

# Define initial parameters for velocity profiling
max_speed_init = 5  # maximum speed (units per second)
min_speed_init = 1  # minimum speed (units per second)
max_accel_init = 1  # maximum acceleration (units per second squared)
max_curvature_init = 10  # maximum allowable curvature for smooth turns

# Step 3: Apply curvature constraint and adjust path
x_smooth, y_smooth = apply_curvature_constraint(x_smooth, y_smooth, max_curvature_init)

# Create the initial velocity profile
velocity_profile = generate_velocity_profile(x_smooth, y_smooth, max_speed_init, min_speed_init, max_curvature_init, max_accel_init)

# Step 4: Set up interactive plot
fig, (ax_path, ax_velocity) = plt.subplots(2, 1, figsize=(10, 6))

# Set up sliders for max_speed, min_speed, and max_curvature
ax_speed_max = plt.axes([0.15, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_speed_min = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_curvature_max = plt.axes([0.15, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_max_speed = Slider(ax_speed_max, 'Max Speed', 0.5, 10.0, valinit=max_speed_init)
slider_min_speed = Slider(ax_speed_min, 'Min Speed', 0.1, 5.0, valinit=min_speed_init)
slider_max_curvature = Slider(ax_curvature_max, 'Max Curvature', 1.0, 20.0, valinit=max_curvature_init)

# Function to update the plot based on the sliders
def update(val):
    max_speed = slider_max_speed.val
    min_speed = slider_min_speed.val
    max_curvature = slider_max_curvature.val
    
    # Apply the curvature constraint and update the path
    global x_smooth, y_smooth  # Ensure x_smooth and y_smooth are updated correctly
    x_smooth, y_smooth = apply_curvature_constraint(x_smooth, y_smooth, max_curvature)
    
    # Generate the new velocity profile
    velocity_profile = generate_velocity_profile(x_smooth, y_smooth, max_speed, min_speed, max_curvature, max_accel_init)
    
    # Update the plot
    plot_path_and_velocity(x_smooth, y_smooth, velocity_profile, selected_points, ax_path, ax_velocity)

# Attach the update function to the sliders
slider_max_speed.on_changed(update)
slider_min_speed.on_changed(update)
slider_max_curvature.on_changed(update)

# Initial plot
plot_path_and_velocity(x_smooth, y_smooth, velocity_profile, selected_points, ax_path, ax_velocity)

# Set up FuncAnimation for live updates
def animate(i):
    update(i)

ani = FuncAnimation(fig, animate, frames=np.arange(0, 100), interval=200, repeat=False)

plt.tight_layout()
plt.show()
