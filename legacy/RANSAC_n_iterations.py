import numpy as np
import matplotlib.pyplot as plt

# Given constants
p = 0.99
n = 5

# Create an array of w values
w_values = np.linspace(0.2, 0.8, 500)

# Compute k for each w using the rearranged equation
k_values = np.log(1 - p) / np.log(1 - w_values**n)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(w_values, k_values, color='blue')
plt.xlabel('Ratio of inlier to outlier points in the contour')
plt.ylabel('Number of iterations of RANSAC required')
# plt.title('Plot of k as a function of w')
plt.grid(True)
plt.legend(title=f'n = {n}, p = {p}')
plt.tight_layout()
plt.show()
