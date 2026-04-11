import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel, ransac  # for fit an ellipse with RANSAC
import cv2  # for the homography
from matplotlib.lines import Line2D  # for custom legend
import scipy.io

"""
Homography transformation (from ellipse to circle) given an ellipse model and the associated regolith interface points.

The script requires:
    - The .mat file containing the parameters of the fitted ellipse model (either in general conic form or in ellipse model form)
    - The .mat file containing the original image (optional, for visualization)

The output includes:
    - The homography matrix that maps the fitted ellipse to a reference circle
    - The sinkage value as a percentage of the wheel radius
    - Plots showing the original image with the fitted ellipse and regolith interface, and the transformed image with the reference circle and transformed regolith interface.

This is accomplished by the following steps:

1. Load the ellipse model parameters and the regolith interface points from .mat files.
2. Compute the homography matrix that maps the fitted ellipse to a reference circle.
3. Apply the homography transformation to the original image and to the regolith interface points.
4. Compute the sinkage value based on the transformed regolith interface and the reference circle. 
5. Visualize the results by plotting the original and transformed images, the fitted ellipse, the reference circle, and the regolith interface points.
"""

def general_conic_to_ellipse_model(p):
    
    """"
    Convert general conic parameters to ellipse model parameters (center, axes, rotation).

    Input:
        - p = [A, B, C, D, E, F] are the coefficients of the general conic equation Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
        
    Output: 
        - (xc, yc): center of the ellipse
        - a: semi-major axis
        - b: semi-minor axis
        - theta: rotation angle of the ellipse (in radians)

    """

    A, B, C, D, E, F = p

    # Center of the ellipse
    den = 4*A*C - B*B
    x0 = (B*E - 2*C*D) / den
    y0 = (B*D - 2*A*E) / den

    # Shifted constant term
    Fp = (A*x0*x0 + B*x0*y0 + C*y0*y0 +
          D*x0 + E*y0 + F)

    # Eigen decomposition of quadratic form
    M = np.array([[A, B/2], [B/2, C]])
    eigvals, eigvecs = np.linalg.eigh(M)

    # Semi-major and semi-minor axes
    a = np.sqrt(-Fp / eigvals[0])
    b = np.sqrt(-Fp / eigvals[1])
    if a < b:
        a, b = b, a
        eigvecs = eigvecs[:, ::-1]

    # Orientation of the ellipse
    theta = np.arctan2(eigvecs[1,0], eigvecs[0,0])

    return x0, y0, a, b, theta


def homography_from_ellipse(
    xc_ransac: float,
    yc_ransac: float,
    a_ransac: float,
    b_ransac: float,
    theta_ransac: float,
    shift: float,
    radius_circle: float = 600,
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Compute the homography matrix from the fitted ellipse to a circle from parameters provided by RANSAC.

    Inputs:
        - Ellipse parameters (center (xc_ransac, yc_ransac), axes (a_ransac, b_ransac), rotation (theta_ransac)) 
        - Shift value for better alignment
        - Radius of the reference circle (default 600)

    Output:
        - Homography matrix (3x3) that maps points from the ellipse to the circle
        - Status of the homography estimation (1 if successful, 0 otherwise)

    """

    # Defining reference points belonging to the ellipse
    t = np.linspace(0, 2 * np.pi, 100)
    ellipse_pts = np.column_stack(
        [
            xc_ransac
            + a_ransac * np.cos(t) * np.cos(theta_ransac)
            - b_ransac * np.sin(t) * np.sin(theta_ransac),
            yc_ransac
            + a_ransac * np.cos(t) * np.sin(theta_ransac)
            + b_ransac * np.sin(t) * np.cos(theta_ransac),
        ]
    ).astype(np.float32)

    # Corresponding points on a circle of chosen radius (default 600) centered in the middle of the ellipse
    center_circle = (xc_ransac, yc_ransac)
    circle_pts = np.column_stack(
        [
            center_circle[0] + radius_circle * np.cos(t),
            center_circle[1] + radius_circle * np.sin(t),
        ]
    ).astype(np.float32)

    # Rotating the circle points for better alignment (default shift -31)
    circle_pts = np.roll(circle_pts, shift=shift, axis=0)

    # Compute the homography matrix from ellipse to circle
    h, status = cv2.findHomography(ellipse_pts, circle_pts)

    return h, status

def compute_sinkage(regolith_interface_points, center_circle, radius_circle):
    """
    Compute sinkage as the maximum distance between the reference circle
    and the regolith interface line, measured along the line normal.
    Returns sinkage as a percentage of the circle radius.

    Input:
        - regolith_interface_points: tuple of two points (x1, y1), (x2, y2) defining the regolith interface line in the transformed (circle) space
        - center_circle: tuple (xc, yc) of the center of the reference circle
        - radius_circle: radius of the reference circle

    Output: 
        - sinkage value in percentage of the circle radius
    """

    # Unpacking regoloth interface points and circle parameters
    (x1, y1), (x2, y2) = regolith_interface_points
    xc, yc = center_circle
    R = radius_circle

    # Interface line equation: Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2

    # Perpendicular distance from circle center to line
    d = abs(A*xc + B*yc + C) / np.sqrt(A*A + B*B)

    # Sinkage (penetration depth)
    sinkage = max(0.0, R - d)

    # Convert to percentage of radius
    sinkage_percent = (sinkage / R) * 100.0

    return sinkage_percent


if __name__ == "__main__":

    regolith_points = None

    ################################
    # CHOOSE THE DATA TO BE ANALYZED
    ################################

    ##### Rear wheel JAXA simulant #####
    # mat_file_ellipse = r"results/JAXA simulant, back wheel cam no grousers/JAXA simulant no grousers ellipse model.mat"   # Ellipse model file
    # mat_file = r"Matlab ransacFit/JAXA simulant no grousers.mat"                                                          # Image file
    # shift = 45
    # regolith_points = (0 , 0),(0 , 0)
    # (x1, y1), (x2, y2) = regolith_points

    ##### Rear wheel Moon simulant #####
    # mat_file_ellipse = r"results/Moon simulant, back wheel cam/Moon regolith ellipse model.mat"                           # Ellipse model file
    # mat_file = r"results/Moon simulant, back wheel cam/Moon regolith contour.mat"                                         # Image file

    ##### Front wheel #####
    # mat_file_ellipse = r"results/Front wheel texture segmentation tests/front_high_sinkage_ellipse.mat"                   # Ellipse model file
    # mat_file = r"Matlab ransacFit/front_high_sinkage_with_texture.mat"                                                    # Image file
    # shift = 20

    ##### Front wheel manual tool test #####
    # mat_file_ellipse = r"high_sinkage_manual_tool.mat"                                                                    # Ellipse model file
    # mat_file = r"Matlab ransacFit/front_high_sinkage_with_texture.mat"                                                    # Image file
    # shift = 19

    ###### JAXA 50s SNR test bank, concat ######
    # mat_file_ellipse = r"results/SNR test bank/JAXA/ellipse_model_JAXA_50s_Concat.mat"                                      # Ellipse model file
    # mat_file = r"results/SNR test bank/JAXA/JAXA_50s.mat"                                                                   # Image file    
    # regolith_points = (698 , 1182),(1048 , 920)
    # (x1, y1), (x2, y2) = regolith_points
    # shift = 68
    # # RESULT: 9.2% sinkage

    ##### Moon 10s SNR test bank, concat #####
    mat_file_ellipse = r"results/SNR test bank/Moon/ellipse_moon_10s.mat"
    mat_file = r"results/SNR test bank/Moon/Moon_10s.mat"
    regolith_points = (552.2 , 1202.5),(995.8 , 985.1)
    (x1, y1), (x2, y2) = regolith_points
    shift = 68
    # # RESULT: 12.3% sinkage

    ###### Laser data validation #####
    # # High sinkage
    # mat_file_ellipse = r"results/15012026/ellipse_model_Validation_Highsinkage.mat"
    # mat_file = r"results/15012026/Validation_Highsinkage.mat"
    # regolith_points = (418 , 1166),(1130 , 801)
    # (x1, y1), (x2, y2) = regolith_points
    # shift = 68
    # # Ground truth: Sinkage = 31.0% (ruler)
    # # Result: Sinkage = 34.7%

    # # Low sinkage
    # mat_file_ellipse = r"results/15012026/ellipse_model_Validation_Lowsinkage.mat"
    # mat_file = r"results/15012026/Validation_Lowsinkage.mat"
    # regolith_points = (498 , 1199),(1133 , 790)
    # (x1, y1), (x2, y2) = regolith_points
    # shift = 68
    # # Result: Sinkage = 24.6% 

    ##### 18022026 VALIDATION #####
    # mat_file_ellipse = r"results/ValidationSet/manual_test2_ellipse.mat"
    # mat_file = r"results/ValidationSet/Validation_Test2.mat"
    # regolith_points = (240 , 1197),(753 , 386)
    # (x1, y1), (x2, y2) = regolith_points
    # shift = 71
    # shift = 22
    # # Result: Sinkage = 29.3% (ie. 25.5mm)
    # # Laser ground truth: Sinkage = 33.75% (ie. 29.36mm)

    #########################################
    # EXTRACTING THE IMAGES AND ELLIPSE MODEL
    #########################################

    # Extracting the ellipse model fitted previously
    data_ellipse = scipy.io.loadmat(mat_file_ellipse)
    ellipse_model = data_ellipse['model'][0]
    
    # Extracting the image
    data = scipy.io.loadmat(mat_file)
    img = data['I']

    # Extracting the regolith interface points defined in the ellipse data
    if regolith_points is None:
        regolith_points = data_ellipse['regolith_interface']
        (x1, y1), (x2, y2) = regolith_points

    # Extracting the ellipse parameters
    if len(ellipse_model) == 6:
        # parameters of the general conic equation
        xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac = general_conic_to_ellipse_model(ellipse_model) 
    elif len(ellipse_model) == 5:
         xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac = ellipse_model

    #########################################
    # COMPUTING THE HOMOGRAPHY TRANSFORMATION
    #########################################

    # Do not hesitate to change the value of "shift" for better alignment of the transformed image (ellipse->circle) 
    h, status = homography_from_ellipse(
        xc_ransac, yc_ransac, a_ransac, b_ransac, theta_ransac, shift=shift
    )

    ###################################################################################################
    # APPLYING THE HOMOGRAPHY TRANSFORMATION TO THE ORIGINAL IMAGE AND TO THE REGOLITH INTERFACE POINTS
    ###################################################################################################

    # Define output size for the warped image (same as input image)
    output_size = img.shape[1], img.shape[0]

    # Perform the homography transformation
    img_warped = cv2.warpPerspective(img, h, output_size)

    # Transform the regolith interface points
    regolith_pts_cv = np.asarray(regolith_points, dtype=np.float32).reshape(-1, 1, 2)       # ensure regolith points are in the correct shape
    regolith_pts_h = cv2.perspectiveTransform(regolith_pts_cv, h)                           # apply homography to regolith points               
    (x1_h, y1_h), (x2_h, y2_h) = regolith_pts_h.reshape(2, 2)                               # reshape back to (2,2) for easier handling 


    ############################## 
    # COMPUTING THE SINKAGE VALUE
    ##############################

    sinkage = compute_sinkage(regolith_interface_points=[(x1_h, y1_h), (x2_h, y2_h)], center_circle=(xc_ransac, yc_ransac), radius_circle=600)
    print("Wheel sinkage [% of wheel radius]", sinkage)

    #######################################
    # PREPARING THE PLOTTING OF THE REUSLTS
    #######################################

    ##### Recovering the ellipse points for plotting #####
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

    ###### Reference circle data from the one used to compute the homography #####
    xc, yc = xc_ransac, yc_ransac
    R = 600

    ##### Reference of the radius and interface lines #####

    # Line coefficients Ax + By + C = 0
    A = y2_h - y1_h
    B = x1_h - x2_h
    C = x2_h*y1_h - x1_h*y2_h

    # Projection of center onto line
    den = A*A + B*B
    x_proj = (B*(B*xc - A*yc) - A*C) / den
    y_proj = (A*(-B*xc + A*yc) - B*C) / den

    # Distance from center to line
    d = abs(A*xc + B*yc + C) / np.sqrt(den)

    # Direction from center to projection (unit vector)
    dx = x_proj - xc
    dy = y_proj - yc
    norm = np.sqrt(dx*dx + dy*dy)
    ux = dx / norm
    uy = dy / norm

    # Intersection with circle (radius point along same direction)
    x_circle = xc + ux * R
    y_circle = yc + uy * R

    ##### Custom legend handles #####
    legend_handles = [
        Line2D(
            [],
            [],
            color="red",
            label="Estimated Ellipse RANSAC",
            linewidth=2,
            linestyle="--",
        ),
        # Line2D(
        #     [],
        #     [],
        #     color="green",
        #     label="Reference Circle",
        #     linewidth=2,
        #     linestyle="--",
        # ),
    ]

    legend_handles_solo = [
        Line2D(
            [],
            [],
            color="red",
            label="Reference Circle",
            linewidth=2,
            linestyle="--",
        )
    ]

    ######################################################################
    # THE PLOTTING OF THE REUSLTS WITH THE ORIGINAL AND TRANSFORMED IMAGES
    ######################################################################

    ##### Plot the original image with the fitted ellipse #####
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    # Plot the RANSAC ellipse
    plt.plot(
        ellipse_x,
        ellipse_y,
        color="red",
        label="Estimated Ellipse RANSAC",
        linewidth=2,
        linestyle="--",
    )

    # Plot regolith points and interface line
    plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label='Regolith Interface')         # plot the regolith interface line
    plt.plot([x1, x2], [y1, y2], 'ro', markersize=4)                                    # plot the regolith interface points

    # # Draw reference circle
    # circle_ref = plt.Circle(
    #     (xc_ransac, yc_ransac),
    #     600, 
    #     color="green",
    #     fill=False,
    #     label="Reference Circle",
    #     linewidth=2,
    #     linestyle="--",
    # )
    # plt.gca().add_artist(circle_ref)

    plt.legend(handles=legend_handles)
    plt.axis("off")

    ##### Plot transformed image with reference circle and regolith interface #####
    plt.subplot(1, 2, 2)
    plt.title("Transformed Image (ellipse->circle)")
    plt.imshow(img_warped, cmap='gray')

    # Plotting the reference circle used to compute the homography
    circle_ref = plt.Circle(
        (xc_ransac, yc_ransac),
        R, 
        color="red",
        fill=False,
        label="Reference Circle",
        linewidth=2,
        linestyle="--",
    )
    plt.gca().add_artist(circle_ref)

    # Plot transformed regolith points
    plt.plot([x1_h, x2_h], [y1_h, y2_h], 'r-', linewidth=2, label='Regolith Interface')
    plt.plot([x1_h, x2_h], [y1_h, y2_h], 'ro', markersize=6)

    # # ----- Draw perpendicular line -----
    # plt.plot([xc, x_proj], [yc, y_proj],
    #         color="cyan", linewidth=3, label="Perpendicular to interface")

    # ----- Draw radius line to circle -----
    plt.plot([xc, x_circle], [yc, y_circle],
            color="yellow", linewidth=2, linestyle="--")

    # Mark projection point
    plt.plot(x_proj, y_proj, 'co', markersize=8)

    # ----- Annotate sinkage on the graph -----
    plt.gca().text(
        0.6, 0.5,  # x=right outside axes, y=center
        f"Sinkage = {sinkage:.1f}%",
        color="cyan",
        fontsize=14,
        transform=plt.gca().transAxes,  # use axes coordinates (0..1)
        rotation=0,
        va='center',
        ha='left',
        bbox=dict(facecolor='black', alpha=0.6, pad=5)
    )
    plt.legend(handles=legend_handles_solo)
    plt.axis("off")
    plt.show()

    ################################################################################################################################
    # PLOTTING THE TRANSFORMED IMAGE WITH REFERENCE CIRCLE AND REGOLITH INTERFACE, WITH RADIUS LINE TO CIRCLE AND SINKAGE ANNOTATION
    # Plotting on a separate figure for better visualization of the sinkage annotation and radius line to circle
    ################################################################################################################################

    plt.figure(figsize=(16, 8))
    plt.title("Transformed Image (ellipse->circle)")
    plt.imshow(img_warped, cmap='gray')

    # Plot the reference circle used to compute the homography
    circle_ref = plt.Circle(
        (xc_ransac, yc_ransac),
        600, 
        color="red",
        fill=False,
        label="Reference Circle",
        linewidth=2,
        linestyle="--",
    )

    # Plot transformed regolith points
    plt.plot([x1_h, x2_h], [y1_h, y2_h], 'r-', linewidth=2, label='Regolith Interface')
    plt.plot([x1_h, x2_h], [y1_h, y2_h], 'ro', markersize=6)
    plt.gca().add_artist(circle_ref)

    # ----- Draw perpendicular line -----
    # plt.plot([xc, x_proj], [yc, y_proj],
    #         color="cyan", linewidth=3, label="Perpendicular to interface")

    # ----- Draw radius line to circle -----
    plt.plot([xc, x_circle], [yc, y_circle],
            color="yellow", linewidth=2, linestyle="--")

    # Mark projection point
    plt.plot(x_proj, y_proj, 'co', markersize=8)

    # ----- Annotate sinkage -----
    plt.gca().text(
        0.6, 0.5,  # x=right outside axes, y=center
        f"Sinkage = {sinkage:.1f}%",
        color="cyan",
        fontsize=14,
        transform=plt.gca().transAxes,  # use axes coordinates (0..1)
        rotation=0,
        va='center',
        ha='left',
        bbox=dict(facecolor='black', alpha=0.6, pad=5)
    )
    plt.legend(handles=legend_handles_solo)
    plt.axis("off")
    plt.show()