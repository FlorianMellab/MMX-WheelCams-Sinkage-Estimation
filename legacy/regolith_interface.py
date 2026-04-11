import numpy as np
import matplotlib.pyplot as plt
import scipy.io

#Step 1: Specify the .mat file path
mat_file = r"Matlab ransacFit/Moon regolith contour.mat"  
# mat_file = r"Matlab ransacFit/Gravier on wheel filtered_points.mat"
# mat_file = r"Matlab ransacFit/JAXA simulant no grousers.mat"
# mat_file = r"Matlab ransacFit/Low sinkage front wheel.mat"

#Load the .mat file in Python
data = scipy.io.loadmat(mat_file)

# Extract variables
I = data['I']
x_points = data['x_points'].flatten()
y_points = data['y_points'].flatten()

# Plot all points on the image
plt.figure()
plt.imshow(I, cmap='gray')
plt.scatter(x_points, y_points, c='lightgreen', s=3)
plt.grid(True)
plt.title("Wheel edge points")
plt.show()

# Filter for lower half of the wheel
idx = y_points >= 600
x_points_filtered = x_points[idx]
y_points_filtered = y_points[idx]

# Plot filtered points
plt.figure()
plt.imshow(I, cmap='gray')
plt.scatter(x_points_filtered, y_points_filtered, c='lightgreen', s=3)
plt.grid(True)
plt.title("Filtered points (lower half)")
plt.show()

# Convert to float
x_points_filtered = x_points_filtered.astype(np.float64)
y_points_filtered = y_points_filtered.astype(np.float64)

# Remove duplicates (keep first occurrence, MATLAB style)
_, ia = np.unique(x_points_filtered, return_index=True)
ia = np.sort(ia)  # preserve original order
x_unique = x_points_filtered[ia]
y_unique = y_points_filtered[ia]

# Sort descending
sort_idx = np.argsort(x_unique)[::-1]
x_sorted = x_unique[sort_idx]
y_sorted = y_unique[sort_idx]

# Compute gradients (4-point method)
gradients = np.zeros_like(x_sorted)
n = len(x_sorted)

for i in range(n-3):
    x_segment = x_sorted[i:i+4]
    y_segment = y_sorted[i:i+4]
    p = np.polyfit(x_segment, y_segment, 1)
    gradients[i] = p[0]

# Last 3 points reuse last slope
gradients[-3:] = gradients[-4]

# Plot gradients
plt.figure()
plt.plot(x_sorted, gradients, 'b.-', markersize=8)
plt.xlabel('x')
plt.ylabel('Gradient')
plt.title('Gradient at each point (3-point method)')
plt.grid(True)
plt.show()

# Find first gradient below threshold
threshold = 0.05
first_idx = np.where(np.abs(gradients) < threshold)[0]

if first_idx.size > 0:
    idx = first_idx[0]
    print(f"First gradient below {threshold:.2f} found at x = {x_sorted[idx]:.2f}, y = {y_sorted[idx]:.2f}, gradient = {gradients[idx]:.3f}")
    interaction_point = np.array([x_sorted[idx], y_sorted[idx]])
else:
    print(f"No point found with |gradient| below {threshold:.2f}")
    interaction_point = None

# Count unique x values
unique_count = len(np.unique(x_sorted))
print(f"Unique x values: {unique_count} out of {len(x_sorted)} total points")

# --- Prompt the user to save the point ---
if interaction_point is not None:
    user_input = input("Do you want to save this detected point to the .mat file? (y/n): ").strip().lower()
    if user_input == 'y':
        filename = input("Enter a filename: ").strip()
        if filename == '':
            filename = 'wheel_edge'
        scipy.io.savemat(f'{filename}.mat', {
            'regolith_interface' : interaction_point
                                })
        print(f"Saved as {filename}.mat")
    else:
        print("Result not saved.")