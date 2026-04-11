#Build a manual contour detection tool
#User specifies some points on the contour, ellipse is defined from those points

import numpy as np
import matplotlib.pyplot as plt
from edge_detection_front import read_image
import scipy.io

def point_selection(img):
    print("Select 5 points to define an ellipse")
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap='gray', origin='upper')
    plt.xlabel("X")
    plt.ylabel("Y")
    points = plt.ginput(n=5)
    plt.show()   
    return points


def conic_from_5_points(points):
    if len(points) != 5:
        raise ValueError("Exactly 5 points required.")

    M = []
    for x, y in points:
        M.append([x*x, x*y, y*y, x, y])
    M = np.array(M, dtype=float)

    rhs = -np.ones(5)

    A, B, C, D, E = np.linalg.solve(M, rhs)

    F = 1.0

    return A, B, C, D, E, F

def general_conic_to_ellipse_model(p):
    A, B, C, D, E, F = p

    # Center
    den = 4*A*C - B*B
    x0 = (B*E - 2*C*D) / den
    y0 = (B*D - 2*A*E) / den

    # Shifted constant term
    Fp = (A*x0*x0 + B*x0*y0 + C*y0*y0 +
          D*x0 + E*y0 + F)

    # Eigen decomposition of quadratic form
    M = np.array([[A, B/2], [B/2, C]])
    eigvals, eigvecs = np.linalg.eigh(M)

    # Axes
    a = np.sqrt(-Fp / eigvals[0])
    b = np.sqrt(-Fp / eigvals[1])
    if a < b:
        a, b = b, a
        eigvecs = eigvecs[:, ::-1]

    # Rotation
    theta = np.arctan2(eigvecs[1,0], eigvecs[0,0])

    return x0, y0, a, b, theta

def regolith_interface(img):
    print("Select the points at the regolith interface")
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap='gray', origin='upper')
    plt.xlabel("X")
    plt.ylabel("Y")
    points = plt.ginput(n=2)
    plt.show()  
    return points


if __name__ == "__main__":
    img_path = r"images/18022026_ValidationSet/Extracted Data/Test_2_with_Sinkage_085939_2/images_front/cam_front_000000.png"
    img = read_image(img_path)
    points = point_selection(img)
    print("You selected:", points)

    points_regolith_interface = regolith_interface(img)
    print("You selected:", points_regolith_interface)

    conic_model = conic_from_5_points(points)
    A, B, C, D, E, F = conic_model
    print(conic_model)

    #Plot the RANSAC-fitted ellipse
    height, width = img.shape[:2]
    x_grid, y_grid = np.meshgrid(np.arange(1, width+1), np.arange(1, height+1))

    Z = A * x_grid**2 + B * x_grid * y_grid + C * y_grid**2 + D * x_grid + E * y_grid + F

    #Plot the regolith interface
    (x1, y1), (x2, y2) = points_regolith_interface


    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.contour(x_grid, y_grid, Z, levels=[0], colors='r', linewidths=2)
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
    plt.plot([x1, x2], [y1, y2], 'go', markersize=6)
    plt.title('RANSAC-Fitted Ellipse Overlay')
    plt.show()

    ellipse_model = general_conic_to_ellipse_model(conic_model)

    # Prompt user to save the results
    user_input = input("Do you want to save the ellipse model? (Y/N): ").strip().upper()
    if user_input == 'Y':
        filename = input("Enter a filename: ").strip()
        if filename == '':
            filename = 'wheel_edge'
        scipy.io.savemat(f'{filename}.mat', {
            'model' : ellipse_model,
            'regolith_interface' : points_regolith_interface
                                    })
        print(f"Saved as {filename}.mat")
    else:
        print("Result not saved.")


    