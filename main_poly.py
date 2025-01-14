import matplotlib.pyplot as plt
import numpy as np


# Create an empty graph
plt.figure()
plt.title("select points")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([0 , 640])
plt.ylim([0 , 640])

# ginput to pick points

selected_points = plt.ginput(n=-1, timeout=0)
print("Selected points:", selected_points)


for point in selected_points:
    plt.scatter(point[0], point[1], color='red', label='Selected Point')

x_points, y_points = zip(*selected_points)
poly = np.polyfit(x_points,y_points,4)

print(poly)
plot_poly = np.poly1d(poly)

x_line = np.linspace(0 , 640 , 1000)
y_line = plot_poly(x_line)

plt.plot(x_line,y_line , color = 'blue' , label = 'poly function')
plt.legend()







plt.show()
