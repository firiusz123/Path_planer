import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define the objective function (curvature minimization)
def objective(x, y):
    curvature = calculate_curvature(x, y)
    return np.sum(curvature)  # Minimize total curvature

# Calculate the curvature
def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
    return curvature

# Gradient of the objective function (simplified version)
def gradient(x, y):
    epsilon = 1e-6
    grad_x = np.zeros_like(x)
    grad_y = np.zeros_like(y)
    
    # Calculate the gradient with respect to x and y by numerical differentiation
    for i in range(1, len(x) - 1):
        x_plus = x.copy()
        x_minus = x.copy()
        y_plus = y.copy()
        y_minus = y.copy()
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        y_plus[i] += epsilon
        y_minus[i] -= epsilon
        
        grad_x[i] = (objective(x_plus, y) - objective(x_minus, y)) / (2 * epsilon)
        grad_y[i] = (objective(x, y_plus) - objective(x, y_minus)) / (2 * epsilon)
    
    return grad_x, grad_y

# Gradient Descent for trajectory optimization
def gradient_descent(x, y, learning_rate=0.01, max_iters=1000):
    for i in range(max_iters):
        grad_x, grad_y = gradient(x, y)
        x -= learning_rate * grad_x  # Update x positions
        y -= learning_rate * grad_y  # Update y positions
        
        # Optionally apply curvature constraint to smooth the path
        # x, y = apply_curvature_constraint(x, y, max_curvature=10)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Objective: {objective(x, y)}")
    
    return x, y

# Generate initial trajectory using cubic spline
x_points = np.array([0, 1, 2, 3, 4])
y_points = np.array([0, 2, 0, -2, 0])
cs = CubicSpline(x_points, y_points, bc_type='clamped')

x_smooth = np.linspace(min(x_points), max(x_points), 100)
y_smooth = cs(x_smooth)

# Apply gradient descent to optimize the trajectory
x_optimized, y_optimized = gradient_descent(x_smooth.copy(), y_smooth.copy(), learning_rate=0.01)

# Plot the original and optimized trajectories
plt.figure(figsize=(8, 6))
plt.plot(x_smooth, y_smooth, label="Original Trajectory", color='b')
plt.plot(x_optimized, y_optimized, label="Optimized Trajectory", color='r')
plt.legend()
plt.title("Trajectory Optimization Using Gradient Descent")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
