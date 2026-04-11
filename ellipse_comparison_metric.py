import numpy as np
import scipy
import matplotlib.pyplot as plt
from SNR_classification import percentage_saturated_pixels
import os
from imageio import imread
import skimage

""" 
Ellipse Comparison Metrics

This script computes and visualizes metrics comparing predicted ellipses to ground truth manual ellipses for varying saturation levels. 

The metrics include:
- Symmetric Mean Boundary Distance: Average distance from points on one ellipse to the other, averaged in both directions.
- Hausdorff Distance: Maximum distance from points on one ellipse to the other, capturing worst-case deviation.
- Intersection over Union (IoU): Area of overlap between the two ellipses divided by the area of their union, measuring overall shape similarity. 
Main focus is on Hausdorff distance and IoU, as they provide complementary insights into the quality of the predicted ellipse fit compared to the ground truth.

The script also plots the metrics against the percentage of saturated pixels in the images, with shaded regions indicating acceptable performance thresholds. 
A quadratic curve fit is optionally applied to visualize trends in the data.
"""

# ------------------------------------------------------------
# Ellipse Sampling
# ------------------------------------------------------------

def ellipse_points(cx, cy, a, b, theta, n=2000):
    """
    Sample n boundary points from an ellipse.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Parametric ellipse (unrotated)
    x = a * cos_t
    y = b * sin_t

    # Apply rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    xr = cos_theta * x - sin_theta * y
    yr = sin_theta * x + cos_theta * y

    # Translate
    xr += cx
    yr += cy

    return np.column_stack((xr, yr))


# ------------------------------------------------------------
# Point-to-Ellipse Distance (Accurate)
# ------------------------------------------------------------

def point_to_ellipse_distance(px, py, cx, cy, a, b, theta,
                              tol=1e-12, max_iter=100):
    """
    Compute shortest distance from point (px, py)
    to rotated ellipse (cx, cy, a, b, theta).
    """

    # Transform to ellipse local frame
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)

    dx = px - cx
    dy = py - cy

    x = cos_theta * dx - sin_theta * dy
    y = sin_theta * dx + cos_theta * dy

    # Reflect to first quadrant
    x = abs(x)
    y = abs(y)

    if x == 0 and y == 0:
        return min(a, b)

    # Initial guess
    t = np.arctan2(b * y, a * x)

    for _ in range(max_iter):
        ct = np.cos(t)
        st = np.sin(t)

        f = (a * ct - x) * (-a * st) + (b * st - y) * (b * ct)
        df = (a * st)**2 + (b * ct)**2 \
             + (a * ct - x) * (-a * ct) \
             + (b * st - y) * (-b * st)

        t_new = t - f / df

        if abs(t - t_new) < tol:
            t = t_new
            break

        t = t_new

    xe = a * np.cos(t)
    ye = b * np.sin(t)

    return np.sqrt((xe - x)**2 + (ye - y)**2)


# ------------------------------------------------------------
# Distance Computations
# ------------------------------------------------------------

def one_sided_distances(e1, e2, n=2000):
    """
    Compute distances from sampled boundary points of e1 to ellipse e2.
    """
    pts = ellipse_points(*e1, n=n)

    distances = [
        point_to_ellipse_distance(px, py, *e2)
        for px, py in pts
    ]

    return np.array(distances)


def symmetric_mean_boundary_distance(e1, e2, n=2000):
    d1 = one_sided_distances(e1, e2, n)
    d2 = one_sided_distances(e2, e1, n)
    return 0.5 * (np.mean(d1) + np.mean(d2))


def hausdorff_distance(e1, e2, n=2000):
    """Compute Hausdorff distance between two ellipses."""
    d1 = one_sided_distances(e1, e2, n)
    d2 = one_sided_distances(e2, e1, n)
    return max(np.max(d1), np.max(d2))

def ellipse_iou_raster(e1, e2, image_size=500):
    """
    Compute IoU of two ellipses using rasterization on a grid.
    image_size: size of square grid (pixels)
    """
    # Determine bounding box
    all_centers = np.array([[e1[0], e1[1]], [e2[0], e2[1]]])
    min_x = int(np.floor(all_centers[:,0].min() - max(e1[2], e1[3], e2[2], e2[3]) - 1))
    max_x = int(np.ceil(all_centers[:,0].max() + max(e1[2], e1[3], e2[2], e2[3]) + 1))
    min_y = int(np.floor(all_centers[:,1].min() - max(e1[2], e1[3], e2[2], e2[3]) - 1))
    max_y = int(np.ceil(all_centers[:,1].max() + max(e1[2], e1[3], e2[2], e2[3]) + 1))

    # Grid resolution
    xs = np.linspace(min_x, max_x, image_size)
    ys = np.linspace(min_y, max_y, image_size)
    X, Y = np.meshgrid(xs, ys)

    # Function to rasterize ellipse
    def rasterize(cx, cy, a, b, theta):
        cos_theta = np.cos(-theta)
        sin_theta = np.sin(-theta)
        x_rot = cos_theta * (X - cx) - sin_theta * (Y - cy)
        y_rot = sin_theta * (X - cx) + cos_theta * (Y - cy)
        mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        return mask.astype(np.uint8)

    mask1 = rasterize(*e1)
    mask2 = rasterize(*e2)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union

def plot_ellipse(image_path, ellipse_path, savepath, n_points=200):
    """Visualize the image and overlay the estimated ellipse."""
    # Load .mat file
    data = scipy.io.loadmat(ellipse_path)
    img = skimage.io.imread(image_path)         # Image
    ellipse_model = data["model"]  # Assume shape (5,) = [cx, cy, a, b, theta]

    cx, cy, a, b, theta = ellipse_model.flatten()  # Flatten in case it's 2D
    # Parametric ellipse points
    t = np.linspace(0, 2*np.pi, n_points)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Rotate ellipse
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ellipse_x = cos_theta * x - sin_theta * y + cx
    ellipse_y = sin_theta * x + cos_theta * y + cy

    # Plot image and ellipse
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap="gray", origin="upper")
    plt.plot(
        ellipse_x,
        ellipse_y,
        color="red",
        label="Estimated Ellipse RANSAC",
        linewidth=2,
        linestyle="--",
    )
    plt.axis("equal")
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)    
    plt.show()

    return


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

if __name__ == "__main__":
    # Initialize lists to store metrics
    mean_dist_concat = []
    haus_dist_concat = []
    iou_concat = []

    ################################
    #TO CHANGE FOR DIFFERENT DATASET
    ################################

    # JAXA DATASET
    ########################################################################################
    list = [10, 20, 30, 40, 50, 55, 60]   #List of JAXA images
    folder_path = "images/SNR image bank/JAXA"
    file_path_manual = []
    file_path_auto = []

    for i in list:
        file_path_manual.append(f"results/SNR test bank/JAXA/manual_ellipse_JAXA{i}.mat")
        file_path_auto.append(f"results/SNR test bank/JAXA/ellipse_concat_JAXA{i}.mat")

    x_start = 8      # minimum percentage of saturated pixels to consider for fitting

    ########################################################################################

    # # MOON DATASET
    # ########################################################################################
    # list = [10,15,20,30]                #List of Moon images
    # # folder_path = "images/SNR image bank/Moon"
    # file_path_manual = []
    # file_path_auto = []

    # for i in list:
    #     file_path_manual.append(f"results/SNR test bank/Moon/manual_Moon_{i}s.mat")
    #     file_path_auto.append(f"results/SNR test bank/Moon/ellipse_Moon_{i}s.mat")

    # x_start = 0      # minimum percentage of saturated pixels to consider for fitting

    # ########################################################################################

    for i in range(len(file_path_manual)):

        # JAXA dataset
        file_path = file_path_manual[i]
        gt = scipy.io.loadmat(file_path)

        file_path = file_path_auto[i]
        pred_concat = scipy.io.loadmat(file_path)

        ellipse_gt = gt["model"][0]
        print(ellipse_gt)
        ellipse_pred_concat = pred_concat["model"][0]
        print(ellipse_pred_concat)

        mean_dist_concat.append(symmetric_mean_boundary_distance(ellipse_gt, ellipse_pred_concat))
        haus_dist_concat.append(hausdorff_distance(ellipse_gt, ellipse_pred_concat))
        iou_concat.append(ellipse_iou_raster(ellipse_gt, ellipse_pred_concat))


    list_images = []

    #Extract images from folder    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".tiff")):
            image_path = os.path.join(folder_path, filename)
            list_images.append(image_path)

    list_images.sort()  # keeps plot ordering consistent

    # Compute percentage of saturated pixels for each image
    saturated_percentages = []
    for image_path in list_images:
        img = imread(image_path)
        pct_sat = percentage_saturated_pixels(img)
        print(pct_sat)
        saturated_percentages.append(pct_sat)

    ################################
    # TO CHANGE FOR DIFFRENT METRICS
    ################################

    # # HAUSDORFF DISTANCE
    # #########################################

    # metric = "Hausdorff distance"

    # # Convert to numpy arrays
    # x = np.array(saturated_percentages)
    # y = np.array(haus_dist_concat)
    
    # # Define acceptable region along y-axis
    # y_min = 0      # lower bound of acceptable Hausdorff distance
    # y_max = 400      # upper bound of acceptable Hausdorff distance

    # # Filter out points with Hausdorff distance > 400 for polyfitting
    # mask = y <= 400

    # #############################################

    # IOU
    ########################################

    metric = "Intersection over union"

    # Convert to numpy arrays 
    x = np.array(saturated_percentages)
    y = np.array(iou_concat)

    # Define acceptable region along y-axis
    y_min = 0.6      # lower bound of acceptable IoU
    y_max = 1      # upper bound of acceptable IoU

    # Filter out points with IoU < 0.6
    mask = y >= 0.6

    #############################################
    

    # PLOTTING THE DATA POINT AND ACCEPTABLE REGION
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.xlabel("% of saturated pixels", fontsize=14)
    plt.title('Quality of predicted ellipse compared to ground truth for varying saturation', fontsize=14)
    plt.fill_between(x=[min(x), max(x)], y1=y_min, y2=y_max, color='lightblue', alpha=0.3, label="Acceptable region")       # Shaded acceptable area

    if metric == "Hausdorff distance":
        plt.scatter(saturated_percentages, haus_dist_concat, label='Data points')
        plt.ylabel("Hausdorff distance", fontsize=14)
    elif metric == "Intersection over union":
        plt.scatter(saturated_percentages, iou_concat, label='Data points')
        plt.ylabel("Intersection over union", fontsize=14)
    
    plt.legend(fontsize=14)
    plt.show()

    # FITTING A QUADRATIC CURVE TO THE DATA (OPTIONAL)
    x_filtered = x[mask]
    y_filtered = y[mask]
    print(y)

    # Quadratic fit (degree 2 polynomial)
    coeffs = np.polyfit(x_filtered, y_filtered, 2)
    poly = np.poly1d(coeffs)

    # Generate smooth curve for plotting
    x_fit = np.linspace(x_start, max(x), 500)
    y_fit = poly(x_fit)

    # PLOTTING THE DATA POINTS, FITTED CURVE, AND ACCEPTABLE REGION
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.xlabel("% of saturated pixels", fontsize=14)

    if metric == "Hausdorff distance":
        plt.ylabel("Hausdorff distance", fontsize=14)
    elif metric == "Intersection over union":
        plt.ylabel("Intersection over union", fontsize=14)

    plt.title('Quality of predicted ellipse compared to ground truth for varying SNR', fontsize=14)

    plt.scatter(x, y, label="Data points")
    plt.plot(x_fit, y_fit, "r:",linewidth=2, label="Quadratic fit")
    plt.fill_between(x=[min(x), max(x)], y1=y_min, y2=y_max, color='lightblue', alpha=0.3, label="Acceptable region")       # Shaded acceptable area

    plt.legend()
    plt.show()