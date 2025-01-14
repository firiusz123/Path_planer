import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Function to fit a polynomial using Ridge regression
def ridge_regression(x_cord, y_cord, degree=3, alpha=1.0):
    # Transform the x coordinates into polynomial features
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x_cord.reshape(-1, 1))

    # Apply Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y_cord)
    
    # Return the polynomial function with the fitted coefficients
    return model, poly

# Plot the graph and select points
plt.figure()
plt.title("Select points for ridge regression polynomial fitting")
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

# Fit a polynomial using Ridge regression (degree 3, alpha 1.0)
degree = 3
alpha = 1.0  # Regularization strength (larger alpha = smoother curve)
model, poly = ridge_regression(np.array(x_points), np.array(y_points), degree, alpha)

# Generate x-values for plotting the fitted polynomial curve
x_range = np.linspace(min(x_points), max(x_points), 1000)
X_poly_range = poly.transform(x_range.reshape(-1, 1))
y_poly = model.predict(X_poly_range)

# Plot the fitted polynomial curve
plt.plot(x_range, y_poly, label=f'Ridge Regression (alpha={alpha})', color='blue')
plt.legend()
plt.show()
