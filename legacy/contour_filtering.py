import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the original .mat file
mat_data = scipy.io.loadmat('Gravier on wheel contour.mat')

# Extract and flatten x and y points
x_points = np.ravel(mat_data['x_points'])
y_points = np.ravel(mat_data['y_points'])

# Define the line through (149, 267) and (359, 729)
x1, y1 = 161, 260
x2, y2 = 332, 736
dx = x2 - x1
dy = y2 - y1

# Compute vector from the line's start to each point
vec_x = x_points - x1
vec_y = y_points - y1

# Cross product to determine side of the line
cross_product = dx * vec_y - dy * vec_x

# Create mask for points on the line or to the right
mask = cross_product <= 0

#Plot image with contour and filtering line
image = mat_data['I']
plt.figure(figsize=(10, 6))
plt.imshow(image, cmap='gray', origin='upper')
# Plot kept points (mask is True) in red
plt.plot(x_points, y_points, color='red', linewidth=2)
# Plot removed points (mask is False) in green
plt.plot([x1, x2], [y1, y2], linewidth=2, color="blue", label='Reference Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Filter points
mat_data['x_points'] = x_points[mask]
mat_data['y_points'] = y_points[mask]

# Save the modified data to a new .mat file
scipy.io.savemat('test.mat', mat_data)   

print("Filtered data saved to 'test.mat'")
