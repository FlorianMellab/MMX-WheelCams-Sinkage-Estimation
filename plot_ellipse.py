import numpy as np
import matplotlib.pyplot as plt
import scipy


""""
Plotting the results of ellipse fitting from a .mat file 
"""

if __name__ == "__main__":

    ################################################
    # LOADING THE ELLIPSE MODEL FRON THE .MAT FILES
    ################################################

    # .mat file path
    mat_file = r"results/Moon simulant, back wheel cam/Moon regolith ellipse model.mat"
    data = scipy.io.loadmat(mat_file)

    #Load data
    I = data['I']
    model = data['model']
    inliers = data['inliers']
    outliers = data['outliers']

    # Unpack the ellipse model parameters (assuming model is a list of parameters)
    for i in range(len(model)):
        [xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac] = model[i]

    # Print the estimated ellipse parameters
    print(
        f"Estimated Ellipse : center=({xc_ransac:.2f}, {yc_ransac:.2f}), "
        f"axes=({a_ransac:.2f}, {b_ransac:.2f}), angle={np.rad2deg(theta_ransac):.2f} degrees"
    )

    #######################################################
    # PLOTTING THE FITTED ELLIPSE WITH INLIERS AND OUTLIERS
    #######################################################

    # Plot the results
    _, ax = plt.subplots()
    ax.imshow(I,cmap='gray', origin='upper')

    # Plot inliers and outliers with different colors
    ax.scatter(
        inliers[:,0],
        inliers[:,1],
        color="yellowgreen",
        marker=".",
        label="Inliers RANSAC",
    )
    ax.scatter(
        outliers[:,0],
        outliers[:,1],
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

    # Draw the estimated ellipse
    ax.plot(
        ellipse_x, ellipse_y, color="green", label="Estimated Ellipse RANSAC", linewidth=3
    )

    ax.legend()
    plt.axis("off")
    plt.show()


   