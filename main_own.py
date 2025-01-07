import matplotlib.pyplot as plt
import numpy as np

def get_thrid_point(x1, y1, x2, y2, alfa):
    # first line equation
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    # perpendicular line equation
    a_perp = -1 / a
    Mx = (x2 + x1) / 2
    My = (y2 + y1) / 2
    b_perp = My - a_perp * Mx

    print(f"Midpoint ({Mx}, {My})")

    third_point = a_perp * (Mx + alfa) + b_perp
    return Mx + alfa, third_point

def quadratic_fit(x1, y1, x2, y2, x3, y3):
    A = np.array([[x1**2, x1, 1],
                  [x2**2, x2, 1],
                  [x3**2, x3, 1]])
    b = np.array([y1, y2, y3])

    # Solve the system to find coefficients a, b, c
    a, b, c = np.linalg.solve(A, b)
    return a, b, c

def get_curve(x_cord, y_cord, alfa):
    curves = []
    points_pairs = len(x_cord) - 1
    for i in range(points_pairs):
        x1, y1 = x_cord[i], y_cord[i]
        x2, y2 = x_cord[i + 1], y_cord[i + 1]
        x3, y3 = get_thrid_point(x1, y1, x2, y2, alfa)

        # Get the coefficients for the quadratic fit
        a, b, c = quadratic_fit(x1, y1, x2, y2, x3, y3)

        curves.append((x1, x2, a, b, c))  # Store the x1, x2 range along with the coefficients

    return curves

# Plot the graph and select points
plt.figure()
plt.title("Select points")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([0, 640])
plt.ylim([0, 640])

selected_points = plt.ginput(n=-1, timeout=0)
print("Selected points:", selected_points)

for point in selected_points:
    plt.scatter(point[0], point[1], color='red', label='Selected Point')

x_points, y_points = zip(*selected_points)

# Get the curves for the selected points
alfa = -6  # Change this value to adjust the offset
curves = get_curve(x_points, y_points, alfa)

# Plot the quadratic curves within the region between each pair of points
for i, (x1, x2, a, b, c) in enumerate(curves):
    # Generate x-values between x1 and x2
    x_range = np.linspace(x1, x2, 1000)
    y_curve = a * x_range**2 + b * x_range + c
    plt.plot(x_range, y_curve, label=f"Quadratic {i + 1}")

plt.legend()
plt.show()
