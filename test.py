import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

# Function to fit a smoothing spline
def smoothing_spline(x_cord, y_cord, smoothing_factor=0.5):
    # Create the smoothing spline with a smoothing factor
    spline = UnivariateSpline(x_cord, y_cord, s=smoothing_factor)
    return spline

# Plot the graph and select points
plt.figure()
plt.title("Select points for smoothing spline fitting")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([0, 640])
plt.ylim([0, 640])

# Select multiple points
selected_points = plt.ginput(n=-1, timeout=0)
print("Selected points:", selected_points)

# Plot the selected points
for point in selected_points:
    plt.scatter(point[0], point[1], color='red', label='Selected Point')

# Extract x and y coordinates from the selected points
x_points, y_points = zip(*selected_points)

# Fit a smoothing spline with a specified smoothing factor
smoothing_factor = 0.6  # Adjust this to control the smoothness
spline = smoothing_spline(np.array(x_points), np.array(y_points), smoothing_factor)

# Generate x-values for plotting the fitted spline curve
x_range = np.linspace(min(x_points), max(x_points), 1000)
y_spline = spline(x_range)

# Plot the smoothing spline curve
plt.plot(x_range, y_spline, label=f'Smoothing Spline (s={smoothing_factor})', color='blue')
plt.legend()
plt.show()
