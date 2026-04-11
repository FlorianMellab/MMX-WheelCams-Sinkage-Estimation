import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel, ransac  # for fit an ellipse with RANSAC
import cv2  # for the homography
from matplotlib.lines import Line2D  # for custom legend
import scipy
import skimage

"""
Ellipse fitting to 2D data points from contour detection using RANSAC. 

The script requires:
    - The .mat files containing the x and y coordinates of the detected edge points
    - The background image (optional) for visualization

The output includes:
    - Estimated ellipse parameters (center, axes, angle)
    - Number of inliers and outliers
    - A plot showing the original points, inliers, outliers, and the fitted ellipse
    - Results can be saved to a .mat file upon user confirmation

The ellipse fitting is performed through the folowing steps:
    1. Load the data points from the .mat files and combine them if necessary.
    3. Use RANSAC to fit an ellipse to the data points, identifying inliers and outliers.
    4. Visualize the inliers, outliers, and the fitted ellipse on the background image.

Note: This script was defined to handle the contour of the font wheel issued from piecewise segmentation.
      This is the reason why there are two .mat files to load and combine, each representing one half of the wheel contour.
"""


def fit_ellipse_ransac(
    data: np.ndarray,
    min_samples: int = 5,
    residual_threshold: float = 3,
    max_trials: int = 600,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """
    Fit an ellipse to 2D data using RANSAC method.
    
    Input:
        data: Nx2 array of (x, y) coordinates of the points to fit
        min_samples: Minimum number of data points to fit an ellipse (default: 5)
        residual_threshold: Maximum distance for a data point to be classified as an inlier (default: 3)
        max_trials: Maximum number of iterations for RANSAC (default: 600)

    Output:
        xc_ransac: x-coordinate of the ellipse center
        yc_ransac: y-coordinate of the ellipse center
        a_ransac: length of the major axis
        b_ransac: length of the minor axis
        theta_ransac: rotation angle of the ellipse in radians
        inliers: boolean array indicating which points are inliers

    """

    _ = EllipseModel()

    # Perform RANSAC to fit the ellipse model to the data
    model_robust, inliers = ransac(
        data,
        EllipseModel,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
    )

    xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac = model_robust.params
    return xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac, inliers

if __name__=="__main__":

    ################################################
    # LOADING THE CONTOUR POINTS FRON THE .MAT FILES
    ################################################

    # Load the first .mat file - left half of the wheel contour
    mat_file = r"results/SNR test bank/JAXA/JAXA_50s.mat"
    data = scipy.io.loadmat(mat_file)

    x_points = data['x_points'].squeeze() 
    y_points = data['y_points'].squeeze()
    I = data['I']  # optional background image

    # Load the second .mat file - right half of the wheel contour
    mat_file = r"results/SNR test bank/JAXA/JAXA_50s_right.mat"
    data = scipy.io.loadmat(mat_file)

    ##########################################################################
    # COMBINE THE TWO SETS OF POINTS INTO A SINGLE DATASET FOR ELLIPSE FITTING
    ##########################################################################

    # Shift x-points by 700, placing them back to their original position before concatenating
    x_points_second = data['x_points'].squeeze() +700
    y_points_second = data['y_points'].squeeze() 

    # Combine first and second datasets
    x_points = np.concatenate([x_points, x_points_second])
    y_points = np.concatenate([y_points, y_points_second])

    detections = np.stack((x_points, y_points), axis=1)

    ###################################################################
    # PERFORM RANSAC ELLIPSE FITTING AND POST-PROCESSING OF THE RESULTS
    ###################################################################

    xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac, inliers = fit_ellipse_ransac(
        detections
    )
    
    print(
        f"Estimated Ellipse : center=({xc_ransac:.2f}, {yc_ransac:.2f}), "
        f"axes=({a_ransac:.2f}, {b_ransac:.2f}), angle={np.rad2deg(theta_ransac):.2f} degrees"
    )

    # Count inliers and outliers
    num_inliers = np.sum(inliers)
    num_outliers = np.sum(~inliers)

    print(f"Number of inliers: {num_inliers}")
    print(f"Number of outliers: {num_outliers}")

    ######################################################################################
    # PLOTTING THE ORIGINAL CONTOUR AND THE FINAL FITTED ELLIPSE WITH INLIERS AND OUTLIERS
    ######################################################################################

    # Plot image and contour points before RANSAC
    plt.figure()
    plt.imshow(I, cmap='gray',origin='upper')
    plt.scatter(x_points, y_points, c='lightgreen', s=3)
    plt.grid(True)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Wheel Edge Points Before RANSAC')
    plt.show()

    # Plot the results after RANSAC ellipse fitting
    _, ax = plt.subplots()
    ax.imshow(I,cmap='gray', origin='upper')

    # Plot inliers and outliers with different colors
    ax.scatter(
        detections[inliers, 0],
        detections[inliers, 1],
        color="yellowgreen",
        marker=".",
        label="Inliers RANSAC",
    )
    ax.scatter(
        detections[~inliers, 0],
        detections[~inliers, 1],
        color="red",
        marker=".",
        label="Outliers RANSAC",
    )

    # Draw the estimated ellipse
    t_fit = np.linspace(0, 2 * np.pi, 1000)
    ellipse_x = (
        xc_ransac
        + a_ransac * np.cos(t_fit) * np.cos(theta_ransac)
        - b_ransac * np.sin(t_fit) * np.sin(theta_ransac)
    )
    ellipse_y = (
        yc_ransac
        + a_ransac * np.cos(t_fit) * np.sin(theta_ransac)
        + b_ransac * np.sin(t_fit) * np.cos(theta_ransac)
    )

    ax.plot(
        ellipse_x, ellipse_y, color="green", label="Estimated Ellipse RANSAC", linewidth=3
    )

    ax.legend()
    plt.axis("off")
    plt.show()

    ####################################################
    # SAVING THE ELLIPSE MODEL AND EXPORTING THE RESULTS
    ####################################################

    model = [xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac]

    user_input = input("Do you want to save the ellipse model? (Y/N): ").strip().upper()
    if user_input == 'Y':
        filename = input("Enter a filename: ").strip()
        if filename == '':
            filename = 'wheel_edge'
        scipy.io.savemat(f'{filename}.mat', {
            'I' : I,
            'model' : model,
            'inliers' : detections[inliers],
            'outliers' : detections[~inliers],
                                    })
        print(f"Saved as {filename}.mat")
    else:
        print("Result not saved.")